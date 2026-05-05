# Codebase Review: YeastAnalysisTool

Date: 2026-05-05
Reviewer: Claude Code (claude-sonnet-4-6)

## Overview

YeastAnalysisTool is a scientific image analysis application for detecting and classifying yeast cells in mitosis. It uses neural network segmentation (Mask R-CNN via mrcnn) to identify cell pairs and then calculates fluorescence intensities across multiple channels (DAPI, GFP, mCherry). Results are reviewed through a GUI and exported to CSV.

The core functionality works. The primary concerns are maintainability, testability, and accumulated technical debt from iterative development.

---

## File Summary

| File | Lines | Role |
|---|---|---|
| main.py | 1,325 | Central logic: segmentation, UI display, data model |
| segment_images.py | 634 | Partially refactored duplicate of main.py segmentation |
| stats.py | 334 | Per-cell fluorescence intensity calculations |
| display.py | 288 | Startup configuration GUI |
| export.py | 77 | CSV export |
| opts.py | 29 | Global configuration constants |
| services.py | 6 | One utility function |
| StatPlugin.py | 21 | Abstract base class for stat plugins |
| stats_plugins/MCherry.py | 40 | Plugin implementation (not integrated) |
| config.json | 9 | Runtime configuration (loaded at startup) |

---

## Prioritized Changes

### Priority 1 — Highest Impact, Lowest Risk

These changes improve day-to-day readability and are safe to make incrementally.

#### 1.1 Fix typos in config keys and method names

**Problem:** Typos create confusion and make searching the code unreliable.
- `"kernel_diviation"` in config.json should be `"kernel_deviation"`
- `"useChache"` should be `"useCache"`
- `where_to_diplay()` in StatPlugin.py should be `where_to_display()`

**Effort:** Low. Find/replace across files. Update config.json and all references in display.py and main.py simultaneously.

---

#### 1.2 Replace magic numbers with named constants

**Problem:** Numeric thresholds with no explanation appear throughout stats.py and main.py. Examples:
- `contourArea > 100000` (bounding box filter threshold)
- `0.5 * top_val` (cell confusion threshold)
- `rolling_ball(..., 50, ...)` (background subtraction radius)
- DAPI=channel 1, GFP=channel 2, mCherry=channel 3 (assumed everywhere, never stated)

**Fix:** Define constants at the top of each file or in a shared `constants.py`:
```python
CONTOUR_AREA_MAX = 100_000
CELL_CONFUSION_THRESHOLD = 0.5
ROLLING_BALL_RADIUS = 50
CHANNEL_DAPI = 1
CHANNEL_GFP = 2
CHANNEL_MCHERRY = 3
```

**Effort:** Low to medium. No logic changes, only naming.

---

#### 1.3 Remove dead and commented-out code

**Problem:** Large commented blocks obscure the active code path.
- main.py lines 154-223: commented image loading code
- main.py lines 957-993: commented CFP handling
- stats.py lines 302-309: commented circle/convex hull calculation
- 30+ unresolved TODO comments scattered throughout

**Fix:** Delete commented code. For TODOs, convert actionable ones to GitHub issues and remove the rest. Keep only TODOs that will be addressed imminently.

**Effort:** Low. Pure deletion. Use git history to recover anything if needed.

---

#### 1.4 Replace bare `except` with specific exception types

**Problem:** Silent exception swallowing hides bugs and makes debugging very difficult.
- main.py:446: `except: continue`
- main.py:560: `except: continue`
- stats.py:146: `except: continue`
- segment_images.py:221: `except Exception: continue`

**Fix:** At minimum, log the exception before continuing. Better: identify what exception is expected and catch only that.
```python
# Before
except:
    continue

# After
except ValueError as e:
    print(f"Skipping cell: {e}")
    continue
```

**Effort:** Low to medium. Requires understanding what can fail at each site.

---

### Priority 2 — Significant Debt Reduction

These address structural problems and should follow Priority 1.

#### 2.1 Eliminate duplicate segmentation code

**Problem:** segment_images.py is a partially-refactored version of the segmentation logic in main.py. Both exist, neither is clearly authoritative, and changes to one are not reflected in the other.

**Fix:** Decide which version is correct. Integrate segment_images.py as the single source of truth, import it from main.py, and delete the duplicate code in main.py. This is the highest-value single change for long-term maintainability.

**Effort:** High. Requires careful testing that behavior is preserved.

---

#### 2.2 Consolidate configuration

**Problem:** Configuration is split across three places with unclear precedence:
- `opts.py`: global constants (input/output dirs, flags)
- `config.json`: runtime settings (kernel size, mCherry toggle)
- `pre_config.json` / `default_config.json`: referenced but unclear role
- String `"on"`/`"off"` used as booleans instead of JSON `true`/`false`

**Fix:**
1. Use a single `config.json` with JSON booleans
2. Define a schema or dataclass for the config so all keys are documented and validated at load time
3. Eliminate `opts.py` by moving its constants into config.json or into a `constants.py`

**Effort:** Medium. Affects display.py, main.py, and anywhere opts is imported.

---

#### 2.3 Reduce global state

**Problem:** main.py and display.py use module-level globals that are mutated by many functions. This makes code hard to reason about and impossible to test.

Globals in main.py: `outline_dict`, `image_dict`, `cp_dict`, `current_image`, `current_cell`, `n`
Globals in display.py: `data`, `input_dir`, `output_dir`

**Fix:** Encapsulate related state into a class or pass state explicitly as arguments. A minimal step is grouping related globals into a simple dataclass:
```python
@dataclass
class AppState:
    image_dict: dict
    cp_dict: dict
    current_image: str
    current_cell: int
```

**Effort:** Medium to high. Touches many functions.

---

#### 2.4 Break up main.py

**Problem:** main.py is 1,325 lines and handles three distinct concerns: the data model (CellPair), segmentation logic, and UI display. This makes it hard to navigate and understand.

**Fix:** Split into:
- `models.py`: CellPair class and related data structures
- `segmentation.py`: consolidate with segment_images.py (see 2.1)
- `ui.py` or keep in `main.py`: only the UI display logic

**Effort:** High. Should follow 2.1 and 2.3.

---

### Priority 3 — Architecture and Extensibility

These are longer-term improvements once the above are in place.

#### 3.1 Integrate the plugin system

**Problem:** StatPlugin.py defines an abstract interface, and MCherry.py implements it, but the plugin is not used in the main analysis flow. stats.py contains hardcoded mCherry logic instead.

**Fix:** Wire the plugin system into stats.get_stats() so that plugins are discovered and called automatically. This makes adding new channel analyses much easier.

**Effort:** Medium, assuming models.py and globals are cleaned up first.

---

#### 3.2 Use pathlib.Path for all file paths

**Problem:** File paths are constructed with string concatenation throughout (e.g., `output_dir + '/masks/' + name + '.outline'`). This is fragile across operating systems and harder to read.

**Fix:** Use `pathlib.Path` consistently:
```python
# Before
with open(output_dir + '/masks/' + cp.get_base_name() + '-' + str(cp.id) + '.outline', 'r') as f:

# After
outline_path = Path(output_dir) / 'masks' / f"{cp.get_base_name()}-{cp.id}.outline"
with outline_path.open() as f:
```

**Effort:** Low per change, medium total (many sites).

---

#### 3.3 Add docstrings to public functions

**Problem:** No functions (except abstract methods in StatPlugin.py) have docstrings. The image channel ordering, expected input shapes, and return types are undocumented.

**Fix:** Add one-line or short docstrings to each public function describing inputs, outputs, and any assumptions (e.g., which channel is DAPI).

**Effort:** Medium. Can be done incrementally per file.

---

#### 3.4 Enable unit testing

**Problem:** The current architecture makes automated testing nearly impossible due to global state and hardcoded file I/O.

**Fix:** After addressing 2.1-2.4, extract pure functions (e.g., contour finding, cell pairing logic) with no side effects. These can be tested with synthetic numpy arrays without any file system.

**Effort:** High. Depends on prior refactoring steps.

---

## Summary Table

| # | Change | Impact | Effort | Depends On |
|---|---|---|---|---|
| 1.1 | Fix typos in config/method names | Low | Low | - |
| 1.2 | Replace magic numbers with constants | Medium | Low | - |
| 1.3 | Remove dead and commented code | Medium | Low | - |
| 1.4 | Fix bare except clauses | Medium | Low | - |
| 2.1 | Eliminate duplicate segmentation | High | High | - |
| 2.2 | Consolidate configuration | High | Medium | 1.1 |
| 2.3 | Reduce global state | High | High | - |
| 2.4 | Break up main.py | High | High | 2.1, 2.3 |
| 3.1 | Integrate plugin system | Medium | Medium | 2.3, 2.4 |
| 3.2 | Use pathlib.Path | Low | Medium | - |
| 3.3 | Add docstrings | Medium | Medium | 2.4 |
| 3.4 | Enable unit testing | High | High | 2.1-2.4 |

---

## Known Bugs to Fix Alongside Refactoring

- `segment_images.py:405`: variable `v` referenced but not defined (should be `c_id`)
- `config.json`: string `"on"`/`"off"` values used as booleans cause silent type issues
- `stats.py:282`: file opened without checking existence, will raise unhandled FileNotFoundError
- `main.py:405`: `debug_image` assigned but never used (likely leftover debug code)
- `StatPlugin.py`: `where_to_diplay` typo propagates to any plugin implementations

---

## Recommended Starting Point

Start with **1.1 through 1.4** in a single cleanup PR. These are all safe, low-risk changes that immediately improve readability and will make the larger refactors easier to review. Then tackle **2.1** (deduplication) as the highest-leverage structural change.
