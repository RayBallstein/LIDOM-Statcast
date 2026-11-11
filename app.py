"""LIDOM Statcast Leaderboards - Shiny for Python app.

Run build_data.py first to create the data/ folder, then:
    python -m shiny run --reload app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

DATA = Path(__file__).parent / "data"

# Thresholds
HARD_HIT_MPH = 95           # Hard-Hit%: exit velo >= 95 mph
SWEET_LO, SWEET_HI = 8, 32  # Sweet-Spot%: launch angle 8 to 32 degrees
FASTBALL_CODES = {"FF", "SI", "FT", "FA"}
TOP_EVENTS = 100            # rows shown on single-event leaderboards

# LIDOM logo, embedded so the app needs no extra files
LOGO_B64 = "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"

BALL_COLS = ["date", "game_pk", "batter_id", "player", "team", "pitcher_id",
             "pitcher", "pitcher_team", "result", "exit_velo",
             "launch_angle", "distance"]
PITCH_COLS = ["date", "game_pk", "pitcher_id", "pitcher", "pitcher_team",
              "batter_id", "batter", "batter_team", "pitch_type", "pitch_name",
              "call", "velo", "spin", "swing", "whiff"]


def is_barrel(ev, la):
    """Approximation of the Statcast barrel zone.

    Exit velo >= 98 mph, with a launch angle window that widens as exit
    velocity increases (26-30 degrees at 98 mph).
    """
    return (ev >= 98) & (la <= 50) & (1.5 * ev - la >= 117) & (ev + la >= 124)


# ---------------------------------------------------------------- load data
def read_csv(name, cols):
    path = DATA / name
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path), True
        except Exception:
            pass
    return pd.DataFrame(columns=cols), False


bb, bb_found = read_csv("batted_balls.csv", BALL_COLS)
pit, pit_found = read_csv("pitches.csv", PITCH_COLS)

if not (bb_found and pit_found):
    DATA_MSG = "No data found. Run build_data.py first."
elif not (set(BALL_COLS) <= set(bb.columns) and set(PITCH_COLS) <= set(pit.columns)):
    DATA_MSG = "Data files are from an older version. Re-run build_data.py."
else:
    DATA_MSG = ""

if DATA_MSG:
    bb = pd.DataFrame(columns=BALL_COLS)
    pit = pd.DataFrame(columns=PITCH_COLS)

for c in ("exit_velo", "launch_angle", "distance"):
    bb[c] = pd.to_numeric(bb[c], errors="coerce")
for c in ("velo", "spin", "swing", "whiff"):
    pit[c] = pd.to_numeric(pit[c], errors="coerce")
pit["swing"] = pit["swing"].fillna(0)
pit["whiff"] = pit["whiff"].fillna(0)

# Batted balls with both exit velo and launch angle, so every rate
# uses the same denominator (BBE).
bb = bb.dropna(subset=["exit_velo", "launch_angle"]).copy()
bb["barrel"] = is_barrel(bb["exit_velo"], bb["launch_angle"]).astype(float)
bb["hard_hit"] = (bb["exit_velo"] >= HARD_HIT_MPH).astype(float)
bb["sweet"] = bb["launch_angle"].between(SWEET_LO, SWEET_HI).astype(float)

meta_path = DATA / "meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

TEAMS = ["All teams"] + sorted(
    set(bb["team"].dropna()) | set(bb["pitcher_team"].dropna())
    | set(pit["pitcher_team"].dropna()) | set(pit["batter_team"].dropna())
)

# ------------------------------------------------------------- leaderboards
HIT_BOARDS = {
    "hardest": "Hardest Hit Balls",
    "max_ev": "Max Exit Velocity",
    "longest_hr": "Longest Home Runs",
    "avg_ev": "Average Exit Velocity",
    "ev90": "EV90",
    "whiff": "Whiff%",
    "hard_hit": "Hard-Hit%",
    "barrel": "Barrel%",
    "avg_la": "Average Launch Angle",
    "sweet": "Sweet-Spot%",
}
PIT_BOARDS = {
    "fast_pitches": "Fastest Pitches",
    "max_velo": "Max Pitch Velocity",
    "avg_velo": "Average Pitch Velocity",
    "whiff": "Whiff%",
    "hard_hit": "Hard-Hit%",
    "avg_ev": "Average Exit Velocity",
    "barrel": "Barrel%",
    "max_spin": "Max Fastball Spin",
    "avg_spin": "Average Fastball Spin",
}
LOWER_IS_BETTER = {"hard_hit", "avg_ev", "barrel"}  # pitching boards (allowed)

HIT_SORT = {
    "max_ev": "Max EV", "avg_ev": "Avg EV", "ev90": "EV90",
    "hard_hit": "Hard-Hit%", "barrel": "Barrel%", "avg_la": "Avg LA",
    "sweet": "Sweet-Spot%",
}
PIT_BBE_SORT = {"hard_hit": "Hard-Hit%", "avg_ev": "Avg EV", "barrel": "Barrel%"}

# (label, max, default) for the minimum-sample slider; None = no slider
MIN_CTL = {
    ("hit", "hardest"): None, ("hit", "longest_hr"): None,
    ("hit", "whiff"): ("Minimum swings", 300, 30),
    ("pit", "fast_pitches"): None,
    ("pit", "max_velo"): ("Minimum pitches", 500, 50),
    ("pit", "avg_velo"): ("Minimum pitches", 500, 50),
    ("pit", "whiff"): ("Minimum swings against", 300, 30),
    ("pit", "hard_hit"): ("Minimum batted balls allowed", 100, 15),
    ("pit", "avg_ev"): ("Minimum batted balls allowed", 100, 15),
    ("pit", "barrel"): ("Minimum batted balls allowed", 100, 15),
    ("pit", "max_spin"): ("Minimum fastballs", 300, 25),
    ("pit", "avg_spin"): ("Minimum fastballs", 300, 25),
}
HIT_BBE_CTL = ("Minimum batted balls (BBE)", 100, 15)


def mode_or_first(s):
    m = s.mode()
    return m.iat[0] if len(m) else s.iloc[0]


def add_rank(df):
    df = df.reset_index(drop=True)
    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df


def grid_view(df):
    if DATA_MSG:
        return render.DataGrid(pd.DataFrame({"Message": [DATA_MSG]}))
    return render.DataGrid(df, width="100%", summary=True)


# ------------------------------------------------------------------- UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.panel_conditional(
            "input.side == 'hit'",
            ui.input_select("hit_board", "Hitting Leaderboard", HIT_BOARDS),
        ),
        ui.panel_conditional(
            "input.side == 'pit'",
            ui.input_select("pit_board", "Pitching Leaderboard", PIT_BOARDS),
        ),
        ui.input_select("team", "Team", TEAMS),
        ui.output_ui("min_ctl"),
        ui.input_radio_buttons(
            "order", "Sort order",
            {"desc": "Highest first", "asc": "Lowest first"},
        ),
        ui.download_button("dl", "Download CSV"),
        width=300,
    ),
    ui.tags.img(
        src=f"data:image/png;base64,{LOGO_B64}",
        alt="LIDOM",
        style=(
            "position: fixed; top: 6px; right: 16px; height: 68px; "
            "width: auto; z-index: 1050;"
        ),
    ),
    ui.navset_card_tab(
        ui.nav_panel(
            "Hitting",
            ui.h5(ui.output_text("heading_hit")),
            ui.output_data_frame("grid_hit"),
            value="hit",
        ),
        ui.nav_panel(
            "Pitching",
            ui.h5(ui.output_text("heading_pit")),
            ui.output_data_frame("grid_pit"),
            value="pit",
        ),
        id="side",
    ),
    ui.card(
        ui.card_header("Definitions"),
        ui.markdown(
            f"""
- **BBE**: batted balls with tracked exit velocity and launch angle.
- **EV90**: 90th percentile exit velocity (the speed a hitter reaches or beats on their hardest 10% of batted balls).
- **Hard-Hit%**: share of BBE at {HARD_HIT_MPH}+ mph.
- **Barrel%**: share of BBE in the barrel zone (98+ mph with a matching launch angle window, an approximation of the Statcast definition).
- **Sweet-Spot%**: share of BBE with a launch angle between {SWEET_LO} and {SWEET_HI} degrees.
- **Whiff%**: swings and misses divided by swings. Swings include whiffs, foul balls and balls in play.
- **Pitching Hard-Hit%, Barrel% and Average Exit Velocity** are for batted balls allowed. Lower is better, so they default to lowest first.
- **Fastball spin** uses four-seam, sinker, two-seam and unclassified fastballs, in rpm.
- Tracking coverage can vary by stadium, and early-season samples are small.
            """
        ),
    ),
    title="LIDOM Statcast Leaderboards",
)


# --------------------------------------------------------------- server
def server(input, output, session):

    @reactive.effect
    def _default_order():
        if input.side() == "pit":
            lower = input.pit_board() in LOWER_IS_BETTER
            ui.update_radio_buttons("order", selected="asc" if lower else "desc")
        else:
            ui.update_radio_buttons("order", selected="desc")

    @render.ui
    def min_ctl():
        side = input.side()
        board = input.hit_board() if side == "hit" else input.pit_board()
        cfg = MIN_CTL.get((side, board))
        if side == "hit" and (side, board) not in MIN_CTL:
            cfg = HIT_BBE_CTL
        if cfg is None:
            return ui.TagList()
        label, top, default = cfg
        return ui.input_slider("min_n", label, 1, top, default)

    # -------- hitting
    @reactive.calc
    def table_hit():
        if input.side() != "hit":
            return pd.DataFrame()
        board = input.hit_board()
        team = input.team()
        asc = input.order() == "asc"
        balls = bb if team == "All teams" else bb[bb["team"] == team]

        if board == "hardest":
            out = balls.sort_values("exit_velo", ascending=asc).head(TOP_EVENTS)
            out = out[["player", "team", "exit_velo", "launch_angle",
                       "distance", "result", "pitcher", "date"]].copy()
            out.columns = ["Player", "Team", "Exit Velo", "Launch Angle",
                           "Distance", "Result", "Pitcher", "Date"]
            return add_rank(out.round(1))

        if board == "longest_hr":
            hr = balls[(balls["result"] == "Home Run") & balls["distance"].notna()]
            out = hr.sort_values("distance", ascending=asc).head(TOP_EVENTS)
            out = out[["player", "team", "distance", "exit_velo",
                       "launch_angle", "pitcher", "date"]].copy()
            out.columns = ["Player", "Team", "Distance", "Exit Velo",
                           "Launch Angle", "Pitcher", "Date"]
            return add_rank(out.round(1))

        min_n = input.min_n()

        if board == "whiff":
            p = pit if team == "All teams" else pit[pit["batter_team"] == team]
            out = p.groupby("batter_id").agg(
                Player=("batter", "first"),
                Team=("batter_team", mode_or_first),
                Swings=("swing", "sum"),
                Whiffs=("whiff", "sum"),
            ).reset_index(drop=True)
            out = out[out["Swings"] >= min_n].copy()
            out["Whiff%"] = out["Whiffs"] / out["Swings"] * 100
            out = out.sort_values("Whiff%", ascending=asc)
            return add_rank(out.round(1))

        out = balls.groupby("batter_id").agg(
            Player=("player", "first"),
            Team=("team", mode_or_first),
            BBE=("exit_velo", "size"),
            MaxEV=("exit_velo", "max"),
            AvgEV=("exit_velo", "mean"),
            EV90=("exit_velo", lambda s: s.quantile(0.9)),
            AvgLA=("launch_angle", "mean"),
            HardHit=("hard_hit", "mean"),
            Barrel=("barrel", "mean"),
            Sweet=("sweet", "mean"),
        ).reset_index(drop=True)
        for c in ("HardHit", "Barrel", "Sweet"):
            out[c] = out[c] * 100
        out = out[out["BBE"] >= min_n]
        out.columns = ["Player", "Team", "BBE", "Max EV", "Avg EV", "EV90",
                       "Avg LA", "Hard-Hit%", "Barrel%", "Sweet-Spot%"]
        out = out.sort_values(HIT_SORT[board], ascending=asc)
        return add_rank(out.round(1))

    # -------- pitching
    @reactive.calc
    def table_pit():
        if input.side() != "pit":
            return pd.DataFrame()
        board = input.pit_board()
        team = input.team()
        asc = input.order() == "asc"

        p = pit if team == "All teams" else pit[pit["pitcher_team"] == team]

        if board in ("fast_pitches", "max_velo", "avg_velo"):
            pv = p.dropna(subset=["velo"])

            if board == "fast_pitches":
                out = pv.sort_values("velo", ascending=asc).head(TOP_EVENTS)
                out = out[["pitcher", "pitcher_team", "velo", "pitch_name",
                           "spin", "batter", "date"]].copy()
                out.columns = ["Pitcher", "Team", "Velo", "Pitch Type", "Spin",
                               "Batter", "Date"]
                return add_rank(out.round(1))

            out = pv.groupby("pitcher_id").agg(
                Pitcher=("pitcher", "first"),
                Team=("pitcher_team", mode_or_first),
                Pitches=("velo", "size"),
                MaxVelo=("velo", "max"),
                AvgVelo=("velo", "mean"),
            ).reset_index(drop=True)
            out = out[out["Pitches"] >= input.min_n()]
            out.columns = ["Pitcher", "Team", "Pitches", "Max Velo", "Avg Velo"]
            col = "Max Velo" if board == "max_velo" else "Avg Velo"
            out = out.sort_values(col, ascending=asc)
            return add_rank(out.round(1))

        min_n = input.min_n()

        if board == "whiff":
            out = p.groupby("pitcher_id").agg(
                Pitcher=("pitcher", "first"),
                Team=("pitcher_team", mode_or_first),
                Swings=("swing", "sum"),
                Whiffs=("whiff", "sum"),
            ).reset_index(drop=True)
            out = out[out["Swings"] >= min_n].copy()
            out["Whiff%"] = out["Whiffs"] / out["Swings"] * 100
            out = out.sort_values("Whiff%", ascending=asc)
            return add_rank(out.round(1))

        if board in ("max_spin", "avg_spin"):
            fb = p[p["pitch_type"].isin(FASTBALL_CODES)].dropna(subset=["spin"])
            out = fb.groupby("pitcher_id").agg(
                Pitcher=("pitcher", "first"),
                Team=("pitcher_team", mode_or_first),
                Fastballs=("spin", "size"),
                MaxSpin=("spin", "max"),
                AvgSpin=("spin", "mean"),
            ).reset_index(drop=True)
            out = out[out["Fastballs"] >= min_n]
            out.columns = ["Pitcher", "Team", "Fastballs", "Max Spin", "Avg Spin"]
            col = "Max Spin" if board == "max_spin" else "Avg Spin"
            out = out.sort_values(col, ascending=asc)
            return add_rank(out.round(0))

        # hard_hit, avg_ev, barrel: batted balls allowed
        balls = bb if team == "All teams" else bb[bb["pitcher_team"] == team]
        out = balls.groupby("pitcher_id").agg(
            Pitcher=("pitcher", "first"),
            Team=("pitcher_team", mode_or_first),
            BBE=("exit_velo", "size"),
            AvgEV=("exit_velo", "mean"),
            HardHit=("hard_hit", "mean"),
            Barrel=("barrel", "mean"),
        ).reset_index(drop=True)
        for c in ("HardHit", "Barrel"):
            out[c] = out[c] * 100
        out = out[out["BBE"] >= min_n]
        out.columns = ["Pitcher", "Team", "BBE", "Avg EV", "Hard-Hit%", "Barrel%"]
        out = out.sort_values(PIT_BBE_SORT[board], ascending=asc)
        return add_rank(out.round(1))

    # -------- headings and tables
    def heading(name):
        if not meta:
            return name
        return (f"{name}  -  {meta.get('games', '?')} games | "
                f"{len(bb)} tracked batted balls | {len(pit)} pitches | "
                f"updated {meta.get('updated', '?')}")

    @render.text
    def heading_hit():
        return heading(HIT_BOARDS[input.hit_board()])

    @render.text
    def heading_pit():
        return heading(PIT_BOARDS[input.pit_board()])

    @render.data_frame
    def grid_hit():
        return grid_view(table_hit())

    @render.data_frame
    def grid_pit():
        return grid_view(table_pit())

    @render.download(
        filename=lambda: "lidom_{}_{}.csv".format(
            input.side(),
            input.hit_board() if input.side() == "hit" else input.pit_board(),
        )
    )
    def dl():
        df = table_hit() if input.side() == "hit" else table_pit()
        yield df.to_csv(index=False)


app = App(app_ui, server)
