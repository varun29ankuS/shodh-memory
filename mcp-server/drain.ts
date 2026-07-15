/**
 * Drain controller — keeps the MCP shim alive long enough to finish in-flight
 * tool calls when the host closes stdin on a thread switch (issue #405).
 *
 * Background: MCP hosts such as Claude Desktop close the shim's *stdin* when the
 * user switches threads, but leave *stdout* open. The stdio server transport
 * (`@modelcontextprotocol/sdk`) never listens to stdin "end"/"close"; it writes
 * responses to stdout whenever a request handler's promise resolves. So a tool
 * call that is mid-flight when stdin EOFs can still be answered — as long as the
 * process does not exit first. Previously the shim called `gracefulShutdown` on
 * the stdin "end"/"close" events and exited ~100ms later, abandoning the pending
 * `await` and leaving the caller to eat the host's ~4-minute call timeout.
 *
 * This controller sits between the stdin lifecycle events and shutdown:
 *   - It counts in-flight tool calls (via {@link track}).
 *   - On stdin EOF with nothing in flight, it shuts down immediately (no change).
 *   - On stdin EOF with work in flight, it delays shutdown until every call has
 *     settled and its response has been written, bounded by a grace window so it
 *     can never hang forever.
 *   - If the grace window expires (or stdout is gone), it resolves the abandoned
 *     calls with an explicit error result — written to the still-open stdout —
 *     so the caller gets a prompt error instead of a silent drop, then exits.
 *
 * The controller is transport-agnostic and fully injectable so it can be unit
 * tested without a live process, transport, or backend.
 */

export type TimerHandle = ReturnType<typeof setTimeout>;

export interface DrainControllerOptions<R> {
  /**
   * Upper bound (ms) on how long to wait for in-flight tool calls to settle
   * after stdin EOF before abandoning them. Must exceed the longest a single
   * tool handler can legitimately take, so a slow-but-progressing backend call
   * is never truncated — yet stay well under the host's call timeout so the
   * caller still receives a response (real or {@link abandonResult}).
   */
  graceMs: number;
  /**
   * Result handed to any tool call still in flight when the grace window
   * expires (or stdout is lost). Written to stdout by the normal handler-return
   * path so the caller gets an error rather than nothing.
   */
  abandonResult: R;
  /**
   * Whether the output stream (stdout) can still be written. When false the host
   * is fully gone and no response can be delivered, so draining is pointless and
   * the controller shuts down immediately.
   */
  isOutputWritable: () => boolean;
  /** Perform the real teardown + process exit. Idempotent on the caller's side. */
  shutdown: (reason: string) => void;
  /** Structured log sink; defaults to {@link console.error}. */
  log?: (message: string) => void;
  /** Timer scheduler; injectable for deterministic tests. Defaults to setTimeout. */
  setTimer?: (fn: () => void, ms: number) => TimerHandle;
  /** Timer canceller; injectable for deterministic tests. Defaults to clearTimeout. */
  clearTimer?: (handle: TimerHandle) => void;
}

export class DrainController<R> {
  private inFlight = 0;
  private stdinClosed = false;
  private completed = false;
  private forceAbandon = false;
  private graceTimer: TimerHandle | null = null;

  private readonly graceMs: number;
  private readonly abandonResult: R;
  private readonly isOutputWritable: () => boolean;
  private readonly doShutdown: (reason: string) => void;
  private readonly log: (message: string) => void;
  private readonly setTimer: (fn: () => void, ms: number) => TimerHandle;
  private readonly clearTimer: (handle: TimerHandle) => void;

  // Signal that unblocks every in-flight `track()` race, resolving each one with
  // the abandon result. Triggered exactly once, on forced drain.
  private readonly abandonSignal: Promise<void>;
  private triggerAbandon!: () => void;

  constructor(options: DrainControllerOptions<R>) {
    this.graceMs = options.graceMs;
    this.abandonResult = options.abandonResult;
    this.isOutputWritable = options.isOutputWritable;
    this.doShutdown = options.shutdown;
    this.log = options.log ?? ((message: string) => console.error(message));
    this.setTimer = options.setTimer ?? ((fn, ms) => setTimeout(fn, ms));
    this.clearTimer = options.clearTimer ?? ((handle) => clearTimeout(handle));
    this.abandonSignal = new Promise<void>((resolve) => {
      this.triggerAbandon = resolve;
    });
  }

  /** Number of tool calls currently in flight. */
  get inFlightCount(): number {
    return this.inFlight;
  }

  /** True once stdin has closed and a drain is (or was) pending. */
  get isDraining(): boolean {
    return this.stdinClosed && !this.completed;
  }

  /**
   * Run a tool call under drain tracking. The in-flight count is incremented for
   * the duration and decremented once the work settles (success OR error). If a
   * forced drain occurs while the work is still pending, the returned promise
   * resolves with {@link abandonResult} so the caller is answered promptly; the
   * underlying work is left to finish on its own as the process exits.
   */
  async track<T extends R>(work: () => Promise<T>): Promise<T> {
    this.inFlight++;
    try {
      if (this.forceAbandon) {
        // Grace window already expired before this call even started.
        return this.abandonResult as T;
      }
      // Kick off the work but race it against the abandon signal so a forced
      // drain can pre-empt a hung backend call with a written error result.
      const workPromise = work();
      return await Promise.race([
        workPromise,
        this.abandonSignal.then(() => this.abandonResult as T),
      ]);
    } finally {
      this.inFlight--;
      this.onSettled();
    }
  }

  /**
   * Handle a stdin "end"/"close" event (both fire; the first wins). Decides
   * between immediate shutdown and a bounded drain of in-flight tool calls.
   */
  onStdinClose(reason: string): void {
    if (this.stdinClosed) return; // double-fire guard: end + close both fire
    this.stdinClosed = true;

    if (this.inFlight === 0) {
      // Nothing in flight — behave exactly as before this fix.
      this.shutdown(reason);
      return;
    }

    if (!this.isOutputWritable()) {
      // Host is fully gone; we could not deliver a response even if we waited.
      this.shutdown(
        `${reason} (output stream not writable — cannot deliver ${this.inFlight} in-flight response(s))`,
      );
      return;
    }

    this.log(
      `[shodh-memory] stdin closed with ${this.inFlight} in-flight tool call(s); draining before shutdown (grace ${this.graceMs}ms)...`,
    );
    this.graceTimer = this.setTimer(() => {
      this.forceDrain(
        `drain grace window (${this.graceMs}ms) expired — abandoning ${this.inFlight} in-flight tool call(s) and shutting down`,
      );
    }, this.graceMs);
  }

  /**
   * Handle loss of the output stream (stdout "error"/"close") while draining.
   * Without stdout there is nothing to drain toward, so abandon and exit.
   */
  onOutputLost(reason: string): void {
    if (!this.stdinClosed || this.completed) return;
    this.forceDrain(reason);
  }

  private shutdown(reason: string): void {
    if (this.completed) return;
    this.completed = true;
    if (this.graceTimer !== null) {
      this.clearTimer(this.graceTimer);
      this.graceTimer = null;
    }
    this.doShutdown(reason);
  }

  private onSettled(): void {
    if (this.stdinClosed && this.inFlight === 0) {
      // Last in-flight call finished after stdin EOF. Its response has resolved
      // and will be flushed to stdout by shutdown's own grace period.
      this.shutdown("in-flight tool call(s) drained after stdin EOF — shutting down");
    }
  }

  private forceDrain(reason: string): void {
    this.forceAbandon = true;
    // Unblock in-flight races so each resolves with the abandon result. Only
    // worth attempting if stdout can still carry the error to the caller.
    // Idempotent: triggerAbandon resolves a promise (a no-op if re-called).
    if (this.isOutputWritable()) {
      this.triggerAbandon();
    }
    // `shutdown` is the single completion gate — guards against double-exit.
    this.shutdown(reason);
  }
}
