//! Comprehensive causal/event vocabulary — the shared foundation for the causal
//! spine (OpenIE + CATENA), complementary to the closed GLiREL schema.
//!
//! Grounded in PDTB Contingency.Cause connectives, FrameNet Cause_* / Creating /
//! Giving / Cause_change families, Girju's causal taxonomy (Direct / Preventative /
//! Facilitative / Consequential / Influential), and Causal-TimeBank signals.
//!
//! Two principles the 3-arm investigation proved:
//!   1. **Symmetric.** Causation is not just destruction. `struck/collapsed/killed`
//!      AND `funded/restored/enabled/created/repaired` — a destruction-only schema
//!      silently drops every positive causal chain (a general-product bug).
//!   2. **Concrete > abstract.** Physical/event predicates (struck, collapsed,
//!      closed) are high-precision; abstract/social ones (supported, affected,
//!      influenced) over-fire on co-occurrence and rebuild the hairball. They are
//!      tagged [`is_abstract_social`] so the extractors can precision-gate them.

/// Coarse predicate family (by head-verb lemma). The causal families are the ones
/// that carry the spine; `Comm`/`Own`/`Other` are non-causal context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Direct causation (cause, trigger, lead, result, spark, produce…).
    Causal,
    /// Caused motion (push, propel, carry, drift, veer…) — FrameNet Cause_motion.
    Motion,
    /// Physical impact (strike, ram, collide, crash, topple…).
    Impact,
    /// Cessation / disruption (close, halt, block, suspend, disrupt…).
    Cease,
    /// Harm (kill, injure, destroy, damage, cripple…).
    Harm,
    /// Inchoative collapse (collapse, fail, fall, buckle, sink…).
    Collapse,
    /// Constructive / positive causation (create, build, fund, restore, enable…).
    Create,
    /// Neutral / transformative (change, transform, convert, affect, influence…).
    Change,
    /// Communication (say, report, announce…) — non-causal.
    Comm,
    /// Possession / operation (own, operate, manage…) — non-causal.
    Own,
    /// Unclassified.
    Other,
}

impl Family {
    /// Whether this family carries causal structure (feeds the spine).
    pub fn is_causal(self) -> bool {
        matches!(
            self,
            Family::Causal
                | Family::Motion
                | Family::Impact
                | Family::Cease
                | Family::Harm
                | Family::Collapse
                | Family::Create
                | Family::Change
        )
    }
}

// Family verb lexicons (head-verb lemmas). Comprehensive + symmetric.
const CAUSAL: &[&str] = &[
    "cause",
    "trigger",
    "lead",
    "result",
    "spark",
    "prompt",
    "contribute",
    "produce",
    "generate",
    "create",
    "precipitate",
    "induce",
    "provoke",
    "bring",
    "set",
    "give",
    "drive",
    "force",
    "make",
];
const MOTION: &[&str] = &[
    "move", "push", "propel", "carry", "sweep", "drag", "drift", "veer", "steer", "pin", "shove",
    "blow", "send", "nudge", "pull", "haul",
];
const IMPACT: &[&str] = &[
    "strike", "ram", "collide", "crash", "hit", "slam", "knock", "topple", "sever", "breach",
    "dislodge", "capsize", "ground", "plow", "smash", "rupture",
];
const CEASE: &[&str] = &[
    "close", "halt", "block", "suspend", "shut", "stall", "paralyze", "strand", "disrupt", "stop",
    "end", "cancel", "sever", "cut", "sever", "freeze", "delay",
];
const HARM: &[&str] = &[
    "kill", "injure", "trap", "wound", "destroy", "damage", "cripple", "disable", "wreck", "maim",
    "displace", "evacuate", "hurt", "drown",
];
const COLLAPSE: &[&str] = &[
    "collapse", "fail", "fall", "buckle", "crumble", "sink", "give", "topple",
];
const CREATE: &[&str] = &[
    "create",
    "produce",
    "generate",
    "build",
    "construct",
    "establish",
    "found",
    "launch",
    "form",
    "develop",
    "make",
    "enable",
    "fund",
    "finance",
    "provide",
    "supply",
    "deliver",
    "grant",
    "approve",
    "authorize",
    "award",
    "restore",
    "repair",
    "rebuild",
    "revive",
    "boost",
    "strengthen",
    "reinforce",
    "support",
    "help",
    "increase",
    "raise",
    "grow",
    "expand",
    "improve",
    "enhance",
    "save",
    "pay",
];
const CHANGE: &[&str] = &[
    "change",
    "transform",
    "convert",
    "shift",
    "turn",
    "alter",
    "modify",
    "reshape",
    "affect",
    "influence",
    "shape",
];
const COMM: &[&str] = &[
    "say", "tell", "report", "announce", "describe", "show", "state", "add", "write", "read",
    "call", "confirm", "warn", "note", "claim",
];
const OWN: &[&str] = &[
    "own", "operate", "have", "include", "contain", "belong", "manage", "run",
];

/// Head verbs so semantically light they make empty/noise triples — excluded.
/// Includes the aspectual/existential verbs (`occur`, `happen`) that merely report
/// that an event took place: they are not a distinct event and must not steal the
/// cause/effect slot from the real event they predicate.
const LIGHT_VERBS: &[&str] = &[
    "be", "have", "become", "seem", "remain", "do", "get", "say", "tell", "occur", "happen",
];

/// Abstract/social predicates that OVER-FIRE on mere co-occurrence (measured:
/// `supported` fired 170× spuriously, rebuilding the hairball under a new label).
/// Real but low-precision — extractors should require stronger evidence for these.
const ABSTRACT_SOCIAL: &[&str] = &[
    "support",
    "affect",
    "influence",
    "involve",
    "concern",
    "relate",
    "associate",
    "back",
    "endorse",
    "praise",
    "criticize",
    "oppose",
    "discuss",
    "address",
];

/// Deverbal / nominal EVENT triggers — the inchoative pivots (collapse, blackout,
/// power-loss) live here as NOUNS, so no subject-verb-object extractor ever sees
/// them. This is why the event layer is the spine.
const NOMINAL_EVENTS: &[&str] = &[
    "collapse",
    "collision",
    "allision",
    "blackout",
    "closure",
    "failure",
    "outage",
    "disruption",
    "evacuation",
    "investigation",
    "response",
    "rescue",
    "recovery",
    "spill",
    "damage",
    "destruction",
    "shutdown",
    "grounding",
    "impact",
    "crash",
    "loss",
    "explosion",
    "fire",
    "flooding",
    "delay",
    "suspension",
    "halt",
    "reopening",
    "repair",
    "rebuild",
    "cleanup",
    "search",
    "outcry",
    "shortage",
    "surge",
    "breach",
    "rupture",
    "accident",
    "disaster",
    "emergency",
    "closure",
    "blockage",
];

/// What a signal encodes: causation or pure temporal order. Temporal sequence is
/// NOT causation — `X then Y` orders the events, it does not assert X caused Y
/// (post hoc ergo propter hoc). CATENA keeps them distinct: temporal order can
/// *inform* causal direction (a cause must precede its effect) but never
/// manufactures a causal edge on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkRelation {
    /// Causal: the head clause caused the other.
    Causes,
    /// Temporal: the head clause happened before the other (no causal claim).
    Precedes,
}

/// Which clause is the "head" — the cause (for a causal signal) or the earlier
/// event (for a temporal one).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDir {
    /// Head is the PRECEDING clause (`X caused Y`, `X before Y`, `X then Y`).
    Forward,
    /// Head is the FOLLOWING clause (`Y because of X`, `Y after X`).
    Backward,
}

/// Signal markers as `(marker, direction, relation)`, in STRICT DESCENDING LENGTH
/// order so a longer phrase wins over its prefix (`as a result of` over `as a
/// result`) and the arrow can't invert. Causal markers grounded in PDTB
/// Contingency.Cause; temporal markers in Causal-TimeBank T-SIGNALs.
const SIGNALS: &[(&str, SignalDir, LinkRelation)] = &[
    ("as a result of", SignalDir::Backward, LinkRelation::Causes),
    ("in the wake of", SignalDir::Backward, LinkRelation::Causes),
    ("brought about", SignalDir::Forward, LinkRelation::Causes),
    ("moments after", SignalDir::Backward, LinkRelation::Precedes),
    ("stemming from", SignalDir::Backward, LinkRelation::Causes),
    ("which caused", SignalDir::Forward, LinkRelation::Causes),
    ("gave rise to", SignalDir::Forward, LinkRelation::Causes),
    ("resulting in", SignalDir::Forward, LinkRelation::Causes),
    ("consequently", SignalDir::Forward, LinkRelation::Causes),
    ("subsequently", SignalDir::Forward, LinkRelation::Precedes),
    ("triggered by", SignalDir::Backward, LinkRelation::Causes),
    ("resulted in", SignalDir::Forward, LinkRelation::Causes),
    ("as a result", SignalDir::Forward, LinkRelation::Causes),
    ("followed by", SignalDir::Forward, LinkRelation::Precedes),
    ("leading to", SignalDir::Forward, LinkRelation::Causes),
    ("because of", SignalDir::Backward, LinkRelation::Causes),
    ("triggering", SignalDir::Forward, LinkRelation::Causes),
    ("as soon as", SignalDir::Forward, LinkRelation::Precedes),
    ("following", SignalDir::Backward, LinkRelation::Causes),
    ("therefore", SignalDir::Forward, LinkRelation::Causes),
    ("triggered", SignalDir::Forward, LinkRelation::Causes),
    ("prompting", SignalDir::Forward, LinkRelation::Causes),
    ("thanks to", SignalDir::Backward, LinkRelation::Causes),
    ("owing to", SignalDir::Backward, LinkRelation::Causes),
    ("prior to", SignalDir::Forward, LinkRelation::Precedes),
    ("set off", SignalDir::Forward, LinkRelation::Causes),
    ("thereby", SignalDir::Forward, LinkRelation::Causes),
    ("forcing", SignalDir::Forward, LinkRelation::Causes),
    ("causing", SignalDir::Forward, LinkRelation::Causes),
    ("because", SignalDir::Backward, LinkRelation::Causes),
    ("before", SignalDir::Forward, LinkRelation::Precedes),
    ("led to", SignalDir::Forward, LinkRelation::Causes),
    ("due to", SignalDir::Backward, LinkRelation::Causes),
    ("caused", SignalDir::Forward, LinkRelation::Causes),
    ("after", SignalDir::Backward, LinkRelation::Precedes),
    ("later", SignalDir::Forward, LinkRelation::Precedes),
    ("hence", SignalDir::Forward, LinkRelation::Causes),
    ("amid", SignalDir::Backward, LinkRelation::Causes),
    ("once", SignalDir::Forward, LinkRelation::Precedes),
    ("when", SignalDir::Forward, LinkRelation::Precedes),
    ("then", SignalDir::Forward, LinkRelation::Precedes),
    ("thus", SignalDir::Forward, LinkRelation::Causes),
];

/// Family of a predicate by its head-verb lemma (lowercased). `Other` if unknown.
pub fn family_of(head_verb: &str) -> Family {
    let v = head_verb;
    for (verbs, fam) in [
        (CAUSAL, Family::Causal),
        (IMPACT, Family::Impact),
        (COLLAPSE, Family::Collapse),
        (HARM, Family::Harm),
        (CEASE, Family::Cease),
        (MOTION, Family::Motion),
        (CREATE, Family::Create),
        (CHANGE, Family::Change),
        (COMM, Family::Comm),
        (OWN, Family::Own),
    ] {
        if verbs.contains(&v) {
            return fam;
        }
    }
    Family::Other
}

/// The canonical relation label for a predicate (head-verb lemma). Concrete
/// families map to specific labels; the rest fall back to `Causes`/`RelatedTo`.
pub fn canonical_relation(head_verb: &str) -> &'static str {
    match head_verb {
        "strike" | "ram" | "collide" | "crash" | "hit" | "slam" | "plow" => "Struck",
        "damage" | "destroy" | "cripple" | "disable" | "wreck" | "topple" | "sever" | "breach"
        | "dislodge" | "knock" | "smash" | "rupture" => "Damaged",
        "kill" | "injure" | "wound" | "maim" | "hurt" | "drown" => "Killed",
        "collapse" | "fail" | "fall" | "buckle" | "crumble" | "sink" | "capsize" => "CollapsedInto",
        "close" | "shut" => "Closed",
        "halt" | "stall" | "stop" | "freeze" => "Halted",
        "block" | "strand" => "Blocked",
        "suspend" | "delay" | "cancel" => "Suspended",
        "disrupt" | "paralyze" => "Disrupted",
        "trap" | "displace" | "evacuate" => "Displaced",
        "create" | "produce" | "generate" | "build" | "construct" | "establish" | "found"
        | "launch" | "form" | "develop" => "Created",
        "enable" | "fund" | "finance" | "provide" | "supply" | "deliver" | "grant" | "approve"
        | "authorize" | "award" | "restore" | "repair" | "rebuild" | "revive" | "boost"
        | "strengthen" | "reinforce" | "help" | "save" | "pay" => "Enabled",
        "prevent" | "avert" => "Prevented",
        _ => {
            if family_of(head_verb).is_causal() {
                "Causes"
            } else {
                "RelatedTo"
            }
        }
    }
}

/// Whether `lemma` is a deverbal event trigger (an event even as a bare noun).
pub fn is_nominal_event(lemma: &str) -> bool {
    NOMINAL_EVENTS.contains(&lemma)
}

/// Whether `lemma` is a semantically-light verb to skip as a predicate head.
pub fn is_light_verb(lemma: &str) -> bool {
    LIGHT_VERBS.contains(&lemma)
}

/// Whether `head_verb` is an abstract/social predicate that over-fires on
/// co-occurrence and needs stronger evidence (precision gate).
pub fn is_abstract_social(head_verb: &str) -> bool {
    ABSTRACT_SOCIAL.contains(&head_verb)
}

/// The causal/temporal signal markers with their directions (longest-first).
pub fn signals() -> &'static [(&'static str, SignalDir, LinkRelation)] {
    SIGNALS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_are_symmetric_destruction_and_creation() {
        assert_eq!(family_of("strike"), Family::Impact);
        assert_eq!(family_of("collapse"), Family::Collapse);
        assert_eq!(family_of("kill"), Family::Harm);
        // Positive causation is present and causal — the symmetry fix.
        assert_eq!(family_of("fund"), Family::Create);
        assert_eq!(family_of("restore"), Family::Create);
        assert!(family_of("fund").is_causal());
        assert!(family_of("create").is_causal());
        // Non-causal context families.
        assert_eq!(family_of("announce"), Family::Comm);
        assert!(!family_of("announce").is_causal());
        assert_eq!(family_of("sail"), Family::Other);
    }

    #[test]
    fn canonical_relations_are_specific_where_concrete() {
        assert_eq!(canonical_relation("ram"), "Struck");
        assert_eq!(canonical_relation("destroy"), "Damaged");
        assert_eq!(canonical_relation("kill"), "Killed");
        assert_eq!(canonical_relation("fund"), "Enabled");
        assert_eq!(canonical_relation("create"), "Created");
        assert_eq!(canonical_relation("collapse"), "CollapsedInto");
        // Unknown-but-causal → generic Causes; non-causal → RelatedTo.
        assert_eq!(canonical_relation("trigger"), "Causes");
        assert_eq!(canonical_relation("announce"), "RelatedTo");
    }

    #[test]
    fn nominal_events_and_light_verbs_and_abstract() {
        assert!(is_nominal_event("collapse"));
        assert!(is_nominal_event("blackout"));
        assert!(!is_nominal_event("bridge"));
        assert!(is_light_verb("be") && is_light_verb("say"));
        assert!(!is_light_verb("strike"));
        // "supported" over-fires — flagged for precision gating.
        assert!(is_abstract_social("support"));
        assert!(is_abstract_social("influence"));
        assert!(!is_abstract_social("strike"));
    }

    #[test]
    fn signals_distinguish_causal_from_temporal() {
        let sigs = signals();
        let dir = |s: &str| sigs.iter().find(|(m, _, _)| *m == s).map(|(_, d, _)| *d);
        let rel = |s: &str| sigs.iter().find(|(m, _, _)| *m == s).map(|(_, _, r)| *r);
        // Causal signals are Causes; temporal signals are Precedes — sequence is
        // NOT causation.
        assert_eq!(rel("caused"), Some(LinkRelation::Causes));
        assert_eq!(rel("led to"), Some(LinkRelation::Causes));
        assert_eq!(rel("due to"), Some(LinkRelation::Causes));
        assert_eq!(rel("then"), Some(LinkRelation::Precedes));
        assert_eq!(rel("after"), Some(LinkRelation::Precedes));
        assert_eq!(rel("before"), Some(LinkRelation::Precedes));
        // Direction (the head clause): `X after Y` → Y is the head (earlier).
        assert_eq!(dir("led to"), Some(SignalDir::Forward));
        assert_eq!(dir("due to"), Some(SignalDir::Backward));
        assert_eq!(dir("after"), Some(SignalDir::Backward));
        assert_eq!(dir("before"), Some(SignalDir::Forward));
        // Longer phrases precede their prefixes (matching prefers the specific).
        let pos = |s: &str| sigs.iter().position(|(m, _, _)| *m == s).unwrap();
        assert!(pos("as a result of") < pos("as a result"));
    }
}
