import { feature, mesh } from "topojson-client";
import type { Topology, GeometryCollection } from "topojson-specification";
import { geoContains, type GeoPermissibleObjects } from "d3";
import worldTopology from "@/assets/world-countries-50m.json";
// India's national boundary from the Local Government Directory, via
// bharatlas.com — see india-boundary-LICENSE.txt beside it for provenance, the
// licence, and the extent check performed before adopting it.
import indiaBoundary from "@/assets/india-boundary-lgd.json";

/**
 * The vendored geography, decoded exactly once.
 *
 * Two surfaces draw the world now — the analyst map at /geo and the two dot
 * maps on the briefing — and the decode is not free: 177 country geometries
 * plus the land mesh, re-walked on every mount if each surface owns its own
 * copy. The topology is a constant, so it is decoded at module scope here and
 * both surfaces share the result.
 *
 * NO TILE SERVER, NO NETWORK. Both files are vendored assets inlined into the
 * single index.html the Rust binary embeds; a basemap that fetched anything
 * would break the offline guarantee and quietly leak that someone is looking
 * at a place.
 *
 * THE INDIA BOUNDARY IS NOT A PREFERENCE. Natural Earth draws boundaries on
 * lines of DE-FACTO CONTROL: it splits Jammu & Kashmir, puts Aksai Chin
 * outside the country and treats Arunachal Pradesh as disputed. Shipping that
 * as India's outline is a release blocker for an Indian defence customer, so
 * every surface that draws India draws `INDIA` below — India's own
 * authoritative administrative source, dissolved from the 36 LGD state and UT
 * polygons.
 */

/** The vendored file's `objects` — named so the decode is not `any`. */
type WorldTopology = Topology<{
  countries: GeometryCollection<{ name: string }>;
  land: GeometryCollection;
}>;

const world = worldTopology as unknown as WorldTopology;

/** Every landmass as one filled shape. */
export const LAND = feature(world, world.objects.land) as unknown as GeoPermissibleObjects;

/** Interior borders only — `(a, b) => a !== b` drops the coastline, which LAND
 *  already draws. Drawing both would double-stroke every shore. */
export const BORDERS = mesh(
  world,
  world.objects.countries,
  (a, b) => a !== b,
) as GeoPermissibleObjects;

/** India's official national boundary. See the note above. */
export const INDIA = indiaBoundary as unknown as GeoPermissibleObjects;

/** One country, with the name Natural Earth gives it. */
interface CountryFeature {
  properties: { name: string };
}

const countryCollection = feature(world, world.objects.countries) as unknown as {
  features: (CountryFeature & GeoPermissibleObjects)[];
};

export const COUNTRIES = countryCollection.features;

/**
 * Which country a coordinate falls in, or `null` for open water.
 *
 * A LOOKUP, NOT A GEOCODER. It answers the one question the briefing needs —
 * name the places a corpus's own coordinates already point at — from geometry
 * that is already in the bundle. Nothing is inferred and nothing is fetched:
 * a point either falls inside a polygon or it does not.
 *
 * The 1:110m generalisation is coarse, so a point a few kilometres offshore
 * reads as water. That is the correct failure for this: the alternative is a
 * nearest-country rule, which would confidently name a country for a memory
 * in the middle of the Pacific.
 *
 * Linear over 177 polygons. The caller runs it once per distinct site, and a
 * corpus has tens of those, not thousands.
 */
export function countryAt(lon: number, lat: number): string | null {
  for (const c of COUNTRIES) {
    if (geoContains(c, [lon, lat])) return c.properties.name;
  }
  return null;
}

/** Whether a coordinate falls inside India's official boundary — asked of the
 *  LGD geometry rather than of Natural Earth's, for the reason above. */
export function isInIndia(lon: number, lat: number): boolean {
  return geoContains(INDIA, [lon, lat]);
}
