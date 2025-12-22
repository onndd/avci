
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
    
    # 1. Lag Features (Last 20 results) - Extended for Deep Memory Strategy (v0.5.0)
    for i in range(1, 21):
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
    for threshold in [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 1000.0]:
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

    # 11. Recovery Speed (Toparlanma Hizi) - FIXED (No Lookahead)
    # How fast does it recover after an Instakill (< 1.10)?
    # Logic: Find last instakill, look at the 3 games AFTER it.
    # To be safe: We calculate the 3-game avg at shift(-3) but then shift(4) it forward?
    # Correct Historical Logic:
    # 1. Calculate a rolling mean of size 3. This 'rolling_3_mean' at index T is mean(T, T-1, T-2).
    # 2. We want the mean of T+1, T+2, T+3 relative to the Instakill at T.
    #    So at index T+3, the rolling_mean represents the recovery of the event at T.
    # 3. Identify Instakills.
    # 4. If T-3 was instakill, then T's rolling mean is the 'Recovery Score' for that event.
    
    # Step 1: 3-game moving average (This is purely historical at any point T)
    roll_3_avg = values.shift(1).rolling(3).mean()
    
    # Step 2: Instakill events (< 1.10)
    is_instakill = (values < 1.10)
    
    # Step 3: Align the recovery score to the event.
    # If index (T-3) was instakill, then 'roll_3_avg' at T is the valid recovery score.
    # We create a series where:
    # Value = roll_3_avg IF (is_instakill shifted 3 units back is True)
    # Else = NaN
    # Fix FutureWarning: Ensure explicit type handling
    shifted_instakill = is_instakill.shift(3).fillna(0).astype(bool)
    valid_recovery_score = pd.Series(np.where(shifted_instakill, roll_3_avg, np.nan))
    
    # Step 4: Forward Fill.
    # At any point T, 'last_recovery_score' is the score of the most recent completed sequence.
    df['last_recovery_score'] = valid_recovery_score.ffill().infer_objects(copy=False).fillna(1.0)

    # 12. Fibonacci Distance (DISABLED v0.7.0 - Noise Removal)
    # if 'games_since_10x' in df.columns:
    #     fib_nums = [21, 34, 55, 89, 144, 233, 377, 610, 987]
    #     def get_fib_dist(val):
    #         return min([abs(val - f) for f in fib_nums])
    #     
    #     df['fib_dist_10x'] = df['games_since_10x'].apply(get_fib_dist) 

    # --- NEW v0.7.0: Professional Features ---
    
    # 20. EMA (Exponential Moving Average) - Sensitve Trend
    df['ema_10'] = values.shift(1).ewm(span=10, adjust=False).mean()
    df['ema_25'] = values.shift(1).ewm(span=25, adjust=False).mean()
    df['ema_cross'] = df['ema_10'] - df['ema_25']
    
    # 21. Statistical Moments (Skewness & Kurtosis) - Anomaly Detector
    # Skew: Direction of tail. Kurtosis: Fatness of tail (Probability of extreme events)
    rolled_50 = values.shift(1).rolling(50)
    df['skewness_50'] = rolled_50.skew().fillna(0)
    df['kurtosis_50'] = rolled_50.kurt().fillna(0)
    
    # 22. Volatility Band Position (Bollinger Logic)
    # Normalized position within the 2-std band. 0=Lower, 1=Upper.
    rol_mean_20 = values.shift(1).rolling(20).mean()
    rol_std_20 = values.shift(1).rolling(20).std()
    
    upper_band = rol_mean_20 + (2 * rol_std_20)
    lower_band = rol_mean_20 - (2 * rol_std_20)
    
    df['band_position'] = (values.shift(1) - lower_band) / ((upper_band - lower_band) + 1e-9) 

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

    # --- NEW ADVANCED FEATURES (v0.3.0) ---
    
    # 15. Instakill Radar (Patlama Radari)
    # Counts games since last < 1.05x
    is_instakill_strict = (values < 1.05).astype(int)
    # Group by cumulative sum of instakills to create segments
    # The count within the segment is the 'games since'
    last_insta_idx = pd.Series(np.where(is_instakill_strict, df.index, np.nan)).ffill().shift(1)
    df['time_since_instakill'] = df.index - last_insta_idx
    df['time_since_instakill'] = df['time_since_instakill'].fillna(100) # Default if never seen
    
    # Density in last 50 games
    df['instakill_density_50'] = is_instakill_strict.shift(1).rolling(50).mean().fillna(0)

    # 16. Streak Patterns (Seri Okuma)
    # Encode last 3 games as binary pattern (Win/Loss relative to 2.0x)
    # 1 = >= 2.0x, 0 = < 2.0x
    w = (values >= 2.0).astype(int)
    # Pattern = (w_t-1 * 4) + (w_t-2 * 2) + (w_t-3 * 1)
    # Result -> 0 to 7 (8 unique patterns)
    p1 = w.shift(1).fillna(0)
    p2 = w.shift(2).fillna(0)
    p3 = w.shift(3).fillna(0)
    df['streak_pattern_3'] = (p1 * 4) + (p2 * 2) + (p3 * 1)

    # 17. House Saturation (Virtual Pool)
    # Heuristic: 
    # < 1.50 -> House +1.0
    # > 2.00 -> House - (Result * 0.1)
    # > 10.0 -> House - (Result * 0.5) to drain fast
    inputs = values.shift(1).fillna(0)
    
    def calc_pool_change(x):
        if x < 1.50: return 1.0
        elif x >= 10.0: return -(x * 0.5)
        elif x >= 2.0: return -(x * 0.1)
        return 0.0 # Between 1.50 and 2.0 is neutral
        
    pool_changes = inputs.apply(calc_pool_change)
    # Cumulative Sum to get Pool Size
    raw_pool = pool_changes.cumsum()
    # Normalize: Z-Score over last 200 games to see if it's "High" relative to recent history
    pool_mean = raw_pool.rolling(200).mean()
    pool_std = raw_pool.rolling(200).std()
    df['virtual_pool_score'] = ((raw_pool - pool_mean) / (pool_std + 1e-9)).fillna(0)

    # 18. Macro Cycle (Derin Hafiza) - Limit 500
    # Look back 500 games. Avg gap between 10x.
    # We can do this vectorized by calculating rolling count of 10x in 500 games
    # Then Average Gap = 500 / (Count + 1)
    # Feature = Games_Since_10x / Average_Gap
    
    is_10x = (values >= 10.0).astype(int)
    count_10x_500 = is_10x.shift(1).rolling(500).sum().fillna(0)
    avg_gap_10x = 500 / (count_10x_500 + 1) 
    
    # We already have 'games_since_10x' from feature #6
    if 'games_since_10x' in df.columns:
        df['macro_cycle_10x'] = df['games_since_10x'] / (avg_gap_10x + 1e-9)
        # > 1.0 means "Overdue" relative to local macro history

    # 19. Event Profiling (Olay Tabanli Profilleme) - v0.4.0
    # Logic: "Fingerprint" of 10x, 20x...
    # Profile Features: 'rol_mean_10', 'streak_under_2'
    # We find what these values were JUST BEFORE a high X occurred.
    # Then we check if current values match that profile.
    
    profile_feats = ['rol_mean_10', 'streak_under_2']
    # Ensure these exist
    if all(f in df.columns for f in profile_feats):
        for t in [10.0, 20.0, 30.0, 40.0, 50.0]:
            # Identify rows where the NEXT result was >= T
            # If df['value'][i] >= T, then row [i] contains the features PRECEDING it.
            # (Because features at i are calculated from i-1...0)
            # So we strictly look at rows where value >= T.
            
            is_event = (values >= t)
            
            # Create a Series that has the feature value ONLY if it's an event row, else NaN
            # We treat 'rol_mean_10' as the signature.
            
            # 1. Capture Profile (Expanding Mean of features at Event times)
            # We need to do this for each signature feature.
            # dist = abs(Current_Feat - Historic_Event_Avg_Feat)
            
            total_dist = 0
            for pf in profile_feats:
                feat_val = df[pf]
                # Filter: Keep value only if event happened here
                event_feat_vals = feat_val.where(is_event)
                # Expanding Mean: "Avg value of this feature prior to all T events seen so far"
                hist_profile = event_feat_vals.expanding().mean()
                # Forward Fill: Carry the last known profile forward
                hist_profile = hist_profile.ffill().fillna(0)
                
                # Distance: How far is current state from the 'Ideal Event State'?
                # Normalized distance (simple diff for now)
                dist = (df[pf] - hist_profile).abs()
                total_dist += dist
            
            # Average distance across profile features
            df[f'dist_to_{int(t)}x_profile'] = total_dist / len(profile_feats)


    
    # 23. RSI (Relative Strength Index) - Short Term (7) - REQUESTED v0.8.0 for Low Targets
    # Measures speed and change of price movements.
    delta = values.shift(1).diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=7).mean()
    avg_loss = loss.rolling(window=7).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi_7'] = 100 - (100 / (1 + rs))
    df['rsi_7'] = df['rsi_7'].fillna(50)

    # 24. Streak Pattern 5 (Sequence Matcher) - REQUESTED v0.8.0 for 10x
    # Extending streak_pattern_3 to 5 games to capture longer sequences (L-L-W-L-W)
    # 1 = >= 2.0x, 0 = < 2.0x
    # Pattern = Sum(w_t-k * 2^(k-1)) for k=1..5
    p4 = w.shift(4).fillna(0)
    p5 = w.shift(5).fillna(0)
    df['streak_pattern_5'] = (p1 * 16) + (p2 * 8) + (p3 * 4) + (p4 * 2) + (p5 * 1)

    # 25. Volatility Expansion (Double Speed) - v0.9.3
    # Fast: For 10x-50x (Explosions). Slow: For 2x (Trends).
    
    # Fast (10)
    vol_curr_fast = values.shift(1).rolling(10).std()
    vol_prev_fast = values.shift(2).rolling(10).std()
    df['vol_expansion_fast'] = (vol_curr_fast / (vol_prev_fast + 1e-9)).fillna(1.0)
    
    # Slow (15) - Reverting to v0.9.1 logic for 2x
    vol_curr_slow = values.shift(1).rolling(15).std()
    vol_prev_slow = values.shift(2).rolling(15).std()
    df['vol_expansion_slow'] = (vol_curr_slow / (vol_prev_slow + 1e-9)).fillna(1.0)
    
    # Generic Alias for 'vol_expansion' (used in old reports like v0.9.1) -> Defaults to Fast
    df['vol_expansion'] = df['vol_expansion_fast']
    
    # 26. Gap Consistency (Bosluk Varyansi) - REQUESTED v0.8.0
    # Measures the regularity of 2.0x wins. 
    # High Variance = Chaotic (Don't play). Low Variance = Rhythmic/Safe.
    # Logic: 
    # 1. Identify 2x wins.
    # 2. Calculate 'Time Since Last 2x' (Gap) at every point.
    # 3. Collect the last 5 Gaps.
    # 4. Calculate Std Dev of those 5 gaps.
    
    # We already have streak_under_2 which resets on a win.
    # Actually, we need the gap *between* wins.
    # Let's derive it from 'games_since_2x' derived from the hit mask.
    # Re-using logic from Feature #6 but specifically for gaps.
    
    hit_2x = (values >= 2.0)
    hit_idx = pd.Series(np.where(hit_2x, df.index, np.nan)).ffill()
    # Shift hit_idx to find the Previous hit
    prev_hit_idx = hit_idx.shift(1).where(hit_2x.shift(1)) # Only valid at hit points? No.
    # We need a rolling list of gaps.
    # Simpler approach:
    # 1. Create a Series of Gap sizes (only at Win rows).
    # 2. Reindex to full df (ffill). 
    # 3. Rolling std of that series? No, we need std of the *last 5 gaps* available at time T.
    
    # Valid Gaps:
    # We need the 'games_since_2x' value JUST BEFORE it resets to 0. 
    # That value represents the gap length of the cycle that just finished.
    # Or simply: values >= 2.0. Index diffs.
    idx_2x = df.index[df['value'] >= 2.0]
    gaps_2x = pd.Series(idx_2x, index=idx_2x).diff() # Gap sizes at win indices
    
    # We need to map these gaps back to the main dataframe.
    # At any index T, we want the std dev of the last 5 gaps known up to T.
    # 1. Map gaps to original index (sparse series)
    gap_series = pd.Series(np.nan, index=df.index)
    gap_series.loc[idx_2x] = gaps_2x
    
    # 2. Forward fill the gaps? No, that repeats the same gap.
    # We need a rolling window over the VALID values.
    # Pandas doesn't support "rolling over valid values only" easily on full index.
    # Hack: Calculate rolling std on the sparse series (ignoring NaNs? No, rolling needs contiguous).
    # Correct: Calculate rolling std on the COMPRESSED (only wins) series, then reindex/ffill.
    
    rolling_gap_std = gaps_2x.rolling(window=5).std()
    
    # Align back to main index using forward fill (Because at time T (loss), the market state is defined by the last known gap stats)
    # We shift(1) because we can't see the gap of the *current* win if we are predicting it?
    # Actually if we are predicting outcome at T, we know the history up to T-1.
    # So we take the rolling_gap_std available at the last win, and pull it forward.
    df['gap_std_2x'] = rolling_gap_std.reindex(df.index).ffill().shift(1).fillna(100)
    
    
    # 27. Trap Detector (Tuzak Dedektoru) - REQUESTED v0.8.0
    # Frequency of outcomes ending in .90 to .99 in last 20 games.
    # Already computed 'is_bait' in #8 but that was rolling 150.
    # User requested rolling 20.
    frac_part = values % 1.0
    is_trap = ((frac_part >= 0.90) & (frac_part <= 0.99)).astype(int)
    df['trap_freq_20'] = is_trap.shift(1).rolling(20).mean().fillna(0)

    df = df.dropna()
    return df

def get_model_features(target, all_columns):
    """
    FRANKENSTEIN STRATEGY (v1.0):
    Restoring "Golden Era" feature sets for each target based on historical reports.
    
    2.0x  -> v0.7.0 (The Low-Risk King)
    3.0x  -> v0.9.1 (The Profit Maker)
    5.0x  -> v0.9.1 (Balanced)
    10.0x -> v0.5.0 (Deep Memory Specialist)
    20.0x -> v0.5.0 (Mid-High Specialist)
    30.0x -> v0.9.0 (The Sniper)
    40.0x -> v0.4.0 (The Survivor) / v0.9.0 Hybrid
    50.0x -> v0.6.0 (The Miracle)
    """
    
    # 1. 2.0x: v0.7.0 -> %56.52 Win Rate
    # Focus: Short Lags + Macro Cycle + Stability
    if target == 2.0:
        return [
            "lag_4", "lag_5", "games_since_1000x", "rol_mean_25", 
            "lag_6", "rol_std_25", "rel_str_200", "lag_10", 
            "lag_2", "time_since_instakill",
            "vol_squeeze", "virtual_pool_score" # Supports
        ]

    # 2. 3.0x: v0.9.1 -> 88.00 Net Profit
    # Focus: Lag 5 + Gap Std + Instakill
    elif target == 3.0:
        return [
            "lag_5", "gap_std_2x", "time_since_instakill", "lag_1",
            "rol_mean_15", "rsi_7", "lag_3", "lag_9",
            "bait_density_150", "rol_mean_50"
        ]

    # 3. 5.0x: v0.5.0 was consistent, v0.9.1 was also okay. 
    # Let's use v0.5.0 for safety (Net Profit 5.0 vs v0.9.4 zero)
    # Actually v0.9.1 had 21.00 Net Profit. Let's use v0.9.1 Features.
    elif target == 5.0:
        return [
            "lag_4", "rsi_7", "lag_10", "rol_std_50",
            "last_recovery_score", "vol_expansion", "macro_cycle_10x",
            "rol_mean_15", "rel_str_25", "gap_std_2x"
        ]

    # 4. 10.0x: v0.5.0 -> 303.00 Net Profit (BIG WINNER)
    # Focus: Deep Memory (Lag 18, 19, 20)
    elif target == 10.0:
        return [
            "lag_18", "lag_7", "lag_11", "time_since_instakill",
            "lag_8", "lag_19", "games_since_100x", "lag_20",
            "lag_1", "vol_squeeze", "session_sentiment"
        ]

    # 5. 20.0x: v0.6.0 -> 86.00 Net Profit
    # Focus: Lag 7, 3, 14
    elif target == 20.0:
        return [
            "lag_7", "lag_3", "lag_14", "last_recovery_score",
            "rol_mean_15", "rel_str_25", "vol_squeeze",
            "macro_cycle_10x", "lag_5", "lag_13"
        ]

    # 6. 30.0x: v0.9.0 -> %100 Success (8/8)
    # Focus: Macro Cycle + Gap Std
    elif target == 30.0:
        return [
            "macro_cycle_10x", "lag_5", "gap_std_2x", "rol_mean_50",
            "rol_mean_25", "games_since_10x", "last_recovery_score",
            "rel_str_15", "lag_7", "virtual_pool_score"
        ]

    # 7. 40.0x: v0.4.0 -> %83 Success
    # Focus: Lag 3 + Recovery + Lag 1
    elif target == 40.0:
        return [
            "lag_3", "last_recovery_score", "lag_4", "lag_1",
            "lag_6", "lag_8", "lag_7", "rol_mean_10",
            "lag_5", "dist_to_50x_profile"
        ]

    # 8. 50.0x: v0.6.0 -> 98.00 Net Profit (%100 Win Rate 2/2)
    # Focus: Lag 6, 3, 12, 11 (Cluster Lags)
    elif target == 50.0:
        return [
            "lag_6", "lag_3", "lag_12", "lag_11",
            "lag_5", "lag_17", "rol_mean_15",
            "lag_19", "lag_1", "lag_13"
        ]

    # Default / Fallback
    else:
        return [c for c in all_columns if 'target' not in c and 'id' not in c]

