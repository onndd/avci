
import pandas as pd
import numpy as np
from config_avci import WINDOWS

def extract_features(df, windows=WINDOWS):
    """
    Extracts purely numerical features for LightGBM.
    Focuses on: Moving Averages, Volatility, Streaks, Last N values.
    """
    df = df.copy()
    values = df['value']
    
    # 1. Lag Features (Last 10 results)
    for i in range(1, 11):
        df[f'lag_{i}'] = values.shift(i)
        
    # 2. Rolling Statistics
    for w in windows:
        rolled = values.shift(1).rolling(window=w)
        df[f'rol_mean_{w}'] = rolled.mean()
        df[f'rol_std_{w}'] = rolled.std()
        df[f'rol_max_{w}'] = rolled.max()
        # Relative strength (Are we above or below recent mean?)
        df[f'rel_str_{w}'] = (values.shift(1) - df[f'rol_mean_{w}']) / (df[f'rol_std_{w}'] + 1e-9)

    # 3. Crash / Boom Streaks
    # Streak under 2.0x
    is_under_2 = (values < 2.0).astype(int)
    m = is_under_2.astype(bool)
    df['streak_under_2'] = (m.groupby((~m).cumsum()).cumcount())
    df['streak_under_2'] = df['streak_under_2'].shift(1).fillna(0)
    
    # Streak over 2.0x
    m2 = (values >= 2.0).astype(bool)
    df['streak_over_2'] = (m2.groupby((~m2).cumsum()).cumcount())
    df['streak_over_2'] = df['streak_over_2'].shift(1).fillna(0)

    # --- ADVANCED FEATURES (The Hunter's Logic) ---

    # 4. Volatility Squeeze (The Silence)
    # Ratio of Short-term Volatility (10) to Long-term Volatility (100)
    # < 1.0 means quiet, < 0.5 means VERY quiet (Storm coming)
    vol_short = values.shift(1).rolling(10).std()
    vol_long = values.shift(1).rolling(100).std()
    df['vol_squeeze'] = (vol_short / (vol_long + 1e-9)).fillna(1.0)

    # 5. RTP Gap (House Hoarding)
    # Theory: If recent payout < Long term average, House is 'hoarding'
    # Simple proxies: Global Mean vs Rolling Mean 100
    global_mean = values.shift(1).expanding().mean()
    local_mean = values.shift(1).rolling(100).mean()
    df['rtp_gap'] = (global_mean - local_mean).fillna(0) 
    # Positive = House Hoarding (Global > Local) -> Expect High X
    # Negative = House Bleeding (Global < Local) -> Expect Correction

    # 6. Games Since High X (Time Pressure)
    # Vectorized calculation for 'Games Since' specific multipliers
    for threshold in [10.0, 20.0, 50.0, 100.0, 1000.0]:
        # Create a mask where value >= threshold
        hit_mask = (values >= threshold)
        # Cumulative sum of hits increases by 1 each time hit occurs
        # Group by this sum to isolate segments between hits
        # Cumcount gives count within each segment (0, 1, 2... since last hit)
        # Shift 1 to represent 'entering' the next game
        # Note: We need a slight trick because GroupBy cumcount resets On the hit, not After.
        # Efficient approach:
        last_hit_idx = pd.Series(np.where(hit_mask, df.index, np.nan)).ffill().shift(1)
        df[f'games_since_{int(threshold)}x'] = df.index - last_hit_idx
        df[f'games_since_{int(threshold)}x'] = df[f'games_since_{int(threshold)}x'].fillna(999) # 999 if never seen

    # 7. Momentum Derivative (U-Turn Detect)
    trend_10 = values.shift(1).rolling(10).mean()
    velocity = trend_10.diff()
    acceleration = velocity.diff()
    df['trend_acceleration'] = acceleration.fillna(0)
    
    # --- PSYCHOLOGICAL & PATTERN FEATURES (The 5 Pillars) ---

    # 8. Bait Detector (Tuzak Algilama)
    # Detects frequency of x.90 - x.99 outcomes in the last 150 games
    # Logic: Modulo 1.0 check. if 0.90 <= (val % 1) <= 0.99
    frac_part = values % 1.0
    is_bait = ((frac_part >= 0.90) & (frac_part <= 0.99)).astype(int)
    # Also check specific 'near miss' like 1.9x, 9.8x... but generic fractional checks works well.
    # User asked for 'last 150 games' check.
    df['bait_density_150'] = is_bait.shift(1).rolling(150).mean().fillna(0)

    # 9. Aftershock (Artci Sok)
    # Density of High X (> 5.0) in the last 150 games.
    # Clustering check.
    is_high = (values >= 5.0).astype(int)
    df['high_x_density_150'] = is_high.shift(1).rolling(150).mean().fillna(0)

    # 10. Session Sentiment (Comertlik Endeksi)
    # Rolling Mean (150) vs Expanding Mean
    roll_mean_150 = values.shift(1).rolling(150).mean()
    exp_mean = values.shift(1).expanding().mean()
    df['session_sentiment'] = (roll_mean_150 / (exp_mean + 1e-9)).fillna(1.0)
    # > 1.0: Generous, < 1.0: Stingy

    # 11. Recovery Speed (Toparlanma Hizi)
    # How fast does it recover after an Instakill (< 1.10)?
    # We create a feature representing the "Average of 3 games after the LAST Instakill"
    # This involves finding the index of the last Instakill for every row.
    # Efficient approach:
    # 1. Identify instakills
    is_instakill = (values < 1.10)
    # 2. Get the 'next 3 values' average for every row (lookahead relative to that row)
    #    (shift(-1) + shift(-2) + shift(-3)) / 3
    #    BUT we can only use this if that row is in the PAST.
    #    So we compute 'future_3_avg' for every row, verify it's valid (not NaNs).
    #    Then we essentially want: For current row t, find k < t where is_instakill[k] is True and k is maximized.
    #    Then Feature = future_3_avg[k].
    
    future_3_avg = (values.shift(-1) + values.shift(-2) + values.shift(-3)) / 3
    # Use built-in forward fill matching mechanism
    # Mask: Instakill rows get their 'future_3_avg' val. Others get NaN.
    instakill_recovery_val = pd.Series(np.where(is_instakill, future_3_avg, np.nan))
    # Fill forward: Current row sees the recovery value of the MOST RECENT instakill.
    # Must shift(1) to avoid peaking at current row's potential instakill status (though recovery val is future, 
    # we need the COMPLETED recovery. So strictly we should only show this if k+3 < current_t).
    # For simplicity/speed in this context, we take the last *calculated* recovery.
    # We shift(4) to ensure we are 3 steps past the instakill event to know its outcome?
    # Yes, to be safe against data leakage.
    df['last_recovery_score'] = instakill_recovery_val.shift(4).ffill().fillna(1.0) # Default to 1.0

    # 12. Fibonacci Distance
    # Distance of 'games_since_10x' to nearest Fib number
    if 'games_since_10x' in df.columns:
        fib_nums = [21, 34, 55, 89, 144, 233, 377, 610, 987]
        # Calculate min distance to any fib
        # (This vectorization is slightly tricky, use apply for simplicity or broadcast subtraction)
        def get_fib_dist(val):
            return min([abs(val - f) for f in fib_nums])
        
        df['fib_dist_10x'] = df['games_since_10x'].apply(get_fib_dist) 

    # 13. Max Pain (Umut Isini / Hope Injection)
    # Detects when the market has been "brutal" for a long time (e.g., < 1.20x).
    # Theory: House must release pressure (give a win) to keep players engaged after heavy losses.
    is_pain = (values < 1.20).astype(int)
    # Rolling count of pain in last 20 games
    df['pain_density_20'] = is_pain.shift(1).rolling(20).sum().fillna(0)
    # If density > 15 (75% of games were loss), we are in "Max Pain" zone.

    # 14. Pattern Trap (Simetrik Yanilgi / Anti-Pattern)
    # Detects highly predictable "Zig-Zag" behaviors (Up, Down, Up, Down) which lull players into false rhythm.
    # We calculate how often the direction changes in the last 5 games.
    diffs = values.shift(1).diff()
    direction = np.sign(diffs)
    # Check if direction flip-flopped compared to previous
    is_zigzag = (direction != direction.shift(1)).astype(int)
    # Sum of flips in last 5 games. 
    # 5/5 means perfect Zig-Zag -> High probability that House will BREAK the pattern (Anti-Pattern).
    df['zigzag_density_5'] = is_zigzag.rolling(5).sum().fillna(0)
    
    df = df.dropna()
    return df
