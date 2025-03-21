import pandas as pd
import numpy as np
import talib
from pmdarima import auto_arima
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
import io
import os
import logging

# Thiết lập log lỗi
logging.basicConfig(filename='errors.log', level=logging.ERROR,
                    format='%(asctime)s %(levelname)s:%(message)s')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Hàm kiểm tra và làm sạch dữ liệu
def clean_data(df):
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.sort_values('time')
        df = df.drop_duplicates(subset='time', keep='first')

        df = df.set_index('time')
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(full_date_range)
        df = df.reset_index()
        df.rename(columns={'index': 'time'}, inplace=True)
    else:
        raise ValueError("Cột 'time' không tồn tại trong dữ liệu!")

    numeric_columns = ['close', 'high', 'low', 'open', 'volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if (df[col] < 0).any():
                raise ValueError(f"Cột {col} chứa giá trị âm, không hợp lệ!")
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            if col == 'volume':
                if df['volume'].isnull().any():
                    volume_mean_30 = df['volume'].rolling(window=30, min_periods=1).mean().bfill().ffill()
                    df['volume'] = df['volume'].fillna(volume_mean_30).fillna(0)
            else:
                df[col] = df[col].bfill().ffill()
    return df


# Hàm tính toán các chỉ báo kỹ thuật bằng TA-Lib
def calculate_indicators(df):
    df = clean_data(df)
    
    if 'close' not in df.columns or df['close'].isnull().all():
        raise ValueError("Cột 'close' không tồn tại hoặc chứa toàn giá trị NaN")

    close = df['close'].astype(float).to_numpy()
    high = df['high'].astype(float).to_numpy() if 'high' in df.columns else close
    low = df['low'].astype(float).to_numpy() if 'low' in df.columns else close
    open_price = df['open'].astype(float).to_numpy() if 'open' in df.columns else close
    volume = df['volume'].astype(float).to_numpy() if 'volume' in df.columns else np.ones(len(close))

    indicators = {}
    try:
        if len(close) < 50:
            raise ValueError("Dữ liệu không đủ để tính SMA_50, cần ít nhất 50 hàng")
        if len(close) < 26:
            raise ValueError("Dữ liệu không đủ để tính MACD, cần ít nhất 26 hàng")
        if len(close) < 14:
            raise ValueError("Dữ liệu không đủ để tính RSI_14 hoặc ATR_14, cần ít nhất 14 hàng")

        indicators['SMA_10'] = talib.SMA(close, timeperiod=10)
        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
        indicators['RSI_14'] = talib.RSI(close, timeperiod=14)
        indicators['Bollinger_Upper'], _, indicators['Bollinger_Lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        indicators['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)

        if 'volume' in df.columns and not df['volume'].isnull().all():
            indicators['OBV'] = talib.OBV(close, volume)

        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators.update({'MACD': macd, 'MACD_Signal': macd_signal, 'MACD_Hist': macd_hist})

        for key, value in indicators.items():
            if np.isnan(value).all():
                raise ValueError(f"Chỉ báo {key} trả về toàn giá trị NaN, kiểm tra lại dữ liệu đầu vào")
    except Exception as e:
        raise ValueError(f"Lỗi khi tính toán chỉ báo: {str(e)}")

    df = df.assign(**{k: v for k, v in indicators.items() if v is not None})
    return df

# Hàm dự đoán giá với ARIMA
def predict_price_arima(df, seasonal: bool = None, m_period: int = None):
    df_clean = df.dropna(subset=['close', 'time'])
    if len(df_clean) < 50:
        return None, pd.DataFrame(), None

    close_series = df_clean['close'].astype(float)
    try:
        # Nếu người dùng không cung cấp, tự kiểm tra chu kỳ
        if seasonal is None:
            seasonal = True  # Giả định giá tài chính có tính chu kỳ
        if m_period is None:
            # Gợi ý chu kỳ: tuần (5), tháng (21), chọn theo dữ liệu
            data_length = len(close_series)
            if data_length >= 100:
                m_period = 21  # Chu kỳ tháng
            elif data_length >= 50:
                m_period = 5   # Chu kỳ tuần
            else:
                m_period = 1   # Không chu kỳ rõ ràng

        model = auto_arima(
            close_series,
            seasonal=seasonal,
            m=m_period,
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )
        arima_fit = model.fit(close_series)
        forecast = arima_fit.predict(n_periods=5)

        d = arima_fit.order[1]
        start_idx = max(d, 1)
        if len(close_series) <= start_idx:
            return arima_fit, pd.DataFrame({'Date': [], 'Predicted_Price': []}), None

        train_pred = arima_fit.predict_in_sample(start=start_idx, end=len(close_series) - 1)
        if np.any(close_series[start_idx:] == 0):
            mape = None
        else:
            mape = mean_absolute_percentage_error(close_series[start_idx:], train_pred)


    except Exception as e:
        raise ValueError(f"Lỗi khi huấn luyện ARIMA (seasonal={seasonal}, m={m_period}): {str(e)}")

    last_date = df_clean['time'].iloc[-1]
    dates = pd.date_range(start=last_date, periods=6, freq='B')[1:]
    forecast_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Price': forecast.round(2),
    })
    return arima_fit, forecast_df, mape

# Hàm tính xác suất tín hiệu tăng giá
def calculate_signal_probability(df_clean, last_row):
    probability = 0.5
    weights = {'rsi': 0.3, 'macd': 0.3, 'ma': 0.2, 'obv': 0.1, 'bollinger': 0.1}

    rsi_14 = last_row['RSI_14']
    if not pd.isna(rsi_14):
        if rsi_14 < 30:
            probability += weights['rsi'] * 0.8
        elif rsi_14 > 70:
            probability -= weights['rsi'] * 0.8
        elif 50 < rsi_14 <= 70:
            probability += weights['rsi'] * 0.3
        elif 30 <= rsi_14 < 50:
            probability -= weights['rsi'] * 0.3

    macd = last_row['MACD']
    macd_signal = last_row['MACD_Signal']
    if not pd.isna(macd) and not pd.isna(macd_signal):
        if macd > macd_signal and macd > 0:
            probability += weights['macd'] * 0.7
        elif macd < macd_signal and macd < 0:
            probability -= weights['macd'] * 0.7

    ma20 = last_row['SMA_20']
    ma10 = last_row['SMA_10']
    if not pd.isna(ma20) and not pd.isna(ma10):
        if ma10 > ma20:
            probability += weights['ma'] * 0.5
        else:
            probability -= weights['ma'] * 0.5

    if 'OBV' in df_clean.columns and len(df_clean['OBV'].dropna()) >= 2:
        obv_change = df_clean['OBV'].iloc[-1] - df_clean['OBV'].iloc[-2]
        if obv_change > 0:
            probability += weights['obv'] * 0.4
        elif obv_change < 0:
            probability -= weights['obv'] * 0.4

    bollinger_upper = last_row['Bollinger_Upper']
    bollinger_lower = last_row['Bollinger_Lower']
    current_price = last_row['close']
    if not pd.isna(bollinger_upper) and not pd.isna(bollinger_lower):
        if current_price < bollinger_lower:
            probability += weights['bollinger'] * 0.5
        elif current_price > bollinger_upper:
            probability -= weights['bollinger'] * 0.5

    return max(0.0, min(1.0, probability))

# Hàm huấn luyện mô hình ML
def train_ml_model(df):
    df_clean = df.dropna(subset=['close', 'time'])
    if len(df_clean) < 50:
        return None, None, 0.0, None

    df_clean['Next_Close'] = df_clean['close'].shift(-1)
    df_clean['Price_Increase'] = (df_clean['Next_Close'] > df_clean['close']).astype(int)
    df_clean = df_clean.dropna(subset=['Next_Close', 'Price_Increase', 'RSI_14', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_10', 'ATR_14'])
    if len(df_clean) < 50:
        return None, None, 0.0, None

    features = ['RSI_14', 'MACD', 'MACD_Signal', 'SMA_20', 'SMA_10', 'ATR_14']
    X = df_clean[features]
    y = df_clean['Price_Increase']

    if X.empty or y.empty or len(X) < 20:
        return None, None, 0.0, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    if len(X_train) < 10 or len(X_test) < 2:
        return None, None, 0.0, None

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean() if len(cv_scores) > 0 else 0.0
    return model, scaler, accuracy, features

# Hàm phân tích chi tiết
def generate_detailed_conclusion(df_clean, last_row, current_price):
    rsi_14 = last_row['RSI_14']
    macd = last_row['MACD']
    macd_signal = last_row['MACD_Signal']
    ma20 = last_row['SMA_20']
    ma10 = last_row['SMA_10']
    ma50 = last_row['SMA_50']
    bollinger_upper = last_row['Bollinger_Upper']
    bollinger_lower = last_row['Bollinger_Lower']

    volume_avg_30 = df_clean['volume'].tail(30).mean()
    volume_current = df_clean['volume'].iloc[-1]
    volume_change = ((volume_current - volume_avg_30) / volume_avg_30 * 100) if volume_avg_30 > 0 else 0

    rsi_14_display = str(round(float(rsi_14), 2))
    macd_display = str(round(float(macd), 2))
    macd_signal_display = str(round(float(macd_signal), 2))
    bollinger_upper_display = str(round(float(bollinger_upper), 2))
    bollinger_lower_display = str(round(float(bollinger_lower), 2))
    ma20_display = str(round(float(ma20), 2))
    ma10_display = str(round(float(ma10), 2))
    ma50_display = str(round(float(ma50), 2))

    ma10_diff = np.diff(df_clean['SMA_10'].dropna().tail(5))
    ma20_diff = np.diff(df_clean['SMA_20'].dropna().tail(5))
    ma50_diff = np.diff(df_clean['SMA_50'].dropna().tail(5))
    trend_slope = "tăng" if (np.mean(ma10_diff) > 0 and np.mean(ma20_diff) > 0 and np.mean(ma50_diff) > 0) else "giảm"

    detailed_conclusion = f"""
    <h3>Phân tích chi tiết</h3>
    <p>1. RSI ({rsi_14_display})</p>
    <ul>
        <li>RSI hiện tại ở mức {rsi_14_display}, phản ánh động lực giá.</li>
        <li>{('RSI thấp (< 30), giá có thể đã giảm quá mức.' if float(rsi_14) < 30 else 'RSI trung lập thấp (30-50), thị trường ổn định.' if float(rsi_14) < 50 else 'RSI trung lập cao (50-70), gần quá mua.' if float(rsi_14) < 70 else 'RSI quá mua (70-80), giá tăng quá mức.' if float(rsi_14) < 80 else 'RSI quá mua mạnh (>= 80), nguy cơ điều chỉnh.')}</li>
    </ul>
    <p>2. MACD ({macd_display}, MACD SIGNAL {macd_signal_display})</p>
    <ul>
        <li>MACD {'trên' if float(macd) > float(macd_signal) else 'dưới'} tín hiệu, tạo tín hiệu {'tăng' if float(macd) > float(macd_signal) else 'giảm'}.</li>
        <li>Chênh lệch: {round(float(macd_signal) - float(macd), 2)}.</li>
    </ul>
    <p>3. MA10 ({ma10_display}), MA20 ({ma20_display}), MA50 ({ma50_display})</p>
    <ul>
        <li>Xu hướng dốc: {trend_slope} dựa trên MA10, MA20, MA50.</li>
        <li>Giá ({current_price}) {'trên' if current_price > float(ma10) else 'dưới'} MA10.</li>
    </ul>
    <p>4. Bollinger Bands (Upper {bollinger_upper_display}, Lower {bollinger_lower_display})</p>
    <ul>
        <li>Giá ({current_price}) nằm giữa dải Bollinger.</li>
        <li>Độ rộng: {round(float(bollinger_upper) - float(bollinger_lower), 2)}.</li>
    </ul>
    <p>5. Khối lượng giao dịch</p>
    <ul>
        <li>Khối lượng {'tăng' if volume_change > 0 else 'giảm'} {abs(volume_change):.2f}% so với trung bình 30 ngày.</li>
    </ul>
    """
    return detailed_conclusion

# === Hàm tính chỉ báo kỹ thuật cơ bản ===
# def calculate_indicators(df):
    # df = clean_data(df)

    # close = df['close'].astype(float).to_numpy()
    # high = df['high'].astype(float).to_numpy()
    # low = df['low'].astype(float).to_numpy()
    # volume = df['volume'].astype(float).to_numpy()

    # indicators = {}
    # try:
    #     indicators['SMA_10'] = talib.SMA(close, timeperiod=10)
    #     indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
    #     indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
    #     indicators['RSI_14'] = talib.RSI(close, timeperiod=14)
    #     indicators['Bollinger_Upper'], _, indicators['Bollinger_Lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    #     indicators['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)

    #     indicators['OBV'] = talib.OBV(close, volume)
    #     macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    #     indicators.update({'MACD': macd, 'MACD_Signal': macd_signal, 'MACD_Hist': macd_hist})

    # except Exception as e:
    #     raise ValueError(f"Lỗi khi tính chỉ báo: {str(e)}")

    # df = df.assign(**indicators)
    # return df

# === Hàm mở rộng tính chỉ báo kỹ thuật nâng cao ===
def calculate_extra_indicators(df):
    close = df['close'].astype(float).to_numpy()
    high = df['high'].astype(float).to_numpy()
    low = df['low'].astype(float).to_numpy()
    volume = df['volume'].astype(float).to_numpy()

    extra_indicators = {}
    try:
        extra_indicators['EMA_10'] = talib.EMA(close, timeperiod=10)
        extra_indicators['EMA_20'] = talib.EMA(close, timeperiod=20)
        extra_indicators['EMA_50'] = talib.EMA(close, timeperiod=50)
        extra_indicators['ADX_14'] = talib.ADX(high, low, close, timeperiod=14)

        slowk, slowd = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
        extra_indicators['Stoch_K'] = slowk
        extra_indicators['Stoch_D'] = slowd

        typical_price = (high + low + close) / 3
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
        extra_indicators['VWAP'] = vwap

        mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
        mfv = mfm * volume
        cmf = pd.Series(mfv).rolling(window=20).sum() / pd.Series(volume).rolling(window=20).sum()
        extra_indicators['CMF_20'] = cmf.to_numpy()

        vol_short = pd.Series(volume).rolling(14).mean()
        vol_long = pd.Series(volume).rolling(28).mean()
        vo = ((vol_short - vol_long) / (vol_long + 1e-9)) * 100
        extra_indicators['Volume_Oscillator'] = vo.to_numpy()

        nine_high = pd.Series(high).rolling(9).max()
        nine_low = pd.Series(low).rolling(9).min()
        conv = (nine_high + nine_low) / 2
        period26_high = pd.Series(high).rolling(26).max()
        period26_low = pd.Series(low).rolling(26).min()
        base = (period26_high + period26_low) / 2
        span_a = ((conv + base) / 2).shift(26)
        period52_high = pd.Series(high).rolling(52).max()
        period52_low = pd.Series(low).rolling(52).min()
        span_b = ((period52_high + period52_low) / 2).shift(26)

        extra_indicators['Ichimoku_Conversion'] = conv.to_numpy()
        extra_indicators['Ichimoku_Base'] = base.to_numpy()
        extra_indicators['Ichimoku_Span_A'] = span_a.to_numpy()
        extra_indicators['Ichimoku_Span_B'] = span_b.to_numpy()

    except Exception as e:
        raise ValueError(f"Lỗi tính chỉ báo mở rộng: {str(e)}")

    df = df.assign(**extra_indicators)
    return df


# Hàm phân tích và tư vấn
def analyze_trend(df):
    
    if len(df) < 50:
        return None
    df_clean = calculate_indicators(df)
    df_clean = calculate_extra_indicators(df_clean)
    
    last_row = df_clean.iloc[-1]

    extra_data = {
        'ema10': round(float(last_row['EMA_10']), 2),
        'ema20': round(float(last_row['EMA_20']), 2),
        'ema50': round(float(last_row['EMA_50']), 2),
        'adx14': round(float(last_row['ADX_14']), 2),
        'stoch_k': round(float(last_row['Stoch_K']), 2),
        'stoch_d': round(float(last_row['Stoch_D']), 2),
        'vwap': round(float(last_row['VWAP']), 2),
        'cmf20': round(float(last_row['CMF_20']), 2),
        'volume_osc': round(float(last_row['Volume_Oscillator']), 2),
        'ichimoku_conversion': round(float(last_row['Ichimoku_Conversion']), 2),
        'ichimoku_base': round(float(last_row['Ichimoku_Base']), 2),
        'ichimoku_span_a': round(float(last_row['Ichimoku_Span_A']), 2),
        'ichimoku_span_b': round(float(last_row['Ichimoku_Span_B']), 2),
    }

    current_price = last_row['close']
    rsi_14 = last_row['RSI_14']
    ma20 = last_row['SMA_20']
    ma10 = last_row['SMA_10']
    ma50 = last_row['SMA_50']
    macd = last_row['MACD']
    macd_signal = last_row['MACD_Signal']
    atr_14 = last_row['ATR_14']
    bollinger_upper = last_row['Bollinger_Upper']
    bollinger_lower = last_row['Bollinger_Lower']

    ma10_diff = np.diff(df_clean['SMA_10'].dropna().tail(10)).mean()
    ma20_diff = np.diff(df_clean['SMA_20'].dropna().tail(10)).mean()
    ma50_diff = np.diff(df_clean['SMA_50'].dropna().tail(10)).mean()
    trend_score = 0
    if ma10 > ma20 and ma10_diff > 0:
        trend_score += 1
    if ma20_diff > 0 and ma50_diff > 0:
        trend_score += 1
    if macd > macd_signal and macd > 0:
        trend_score += 1
    if rsi_14 > 50:
        trend_score += 0.5
    trend = "Xu hướng tăng" if trend_score >= 2 else "Xu hướng giảm"

    buy_point = round(bollinger_lower - atr_14 * 0.5, 2)
    sell_point = round(bollinger_upper + atr_14 * 0.5, 2)
    support = round(min(ma10, ma20) - atr_14, 2)
    resistance = round(max(ma10, ma20) + atr_14, 2)

    signal_probability = calculate_signal_probability(df_clean, last_row)
    ml_model, scaler, ml_accuracy, features = train_ml_model(df_clean)
    ml_prediction = None
    if ml_model is not None and scaler is not None:
        last_features = scaler.transform([df_clean[features].iloc[-1].values])
        ml_prediction = ml_model.predict_proba(last_features)[0][1]

    arima_model, forecast_df, arima_mape = predict_price_arima(df_clean)
    if arima_model is None:
        forecast_df = pd.DataFrame()

    candle_pattern = "Không phát hiện mô hình nến nào trong ngày gần nhất."
    if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
        if last_row['close'] > last_row['open'] and last_row['close'] > last_row['high'] * 0.9:
            candle_pattern = "Mô hình nến tăng (Bullish) được phát hiện."
        elif last_row['close'] < last_row['open'] and last_row['close'] < last_row['low'] * 1.1:
            candle_pattern = "Mô hình nến giảm (Bearish) được phát hiện."

    rsi_14_display = str(round(float(rsi_14), 2))
    macd_display = str(round(float(macd), 2))
    macd_signal_display = str(round(float(macd_signal), 2))
    bollinger_upper_display = str(round(float(bollinger_upper), 2))
    bollinger_lower_display = str(round(float(bollinger_lower), 2))
    ma20_display = str(round(float(ma20), 2))
    ma10_display = str(round(float(ma10), 2))
    ma50_display = str(round(float(ma50), 2))

    ai_recommendation = ""
    if ml_prediction is not None:
        ml_prediction_value = round(float(ml_prediction) * 100, 2)
        if ml_prediction_value > 70:
            ai_recommendation = f"📈 Khả năng giá tăng {ml_prediction_value}% trong 3 ngày tới → Khuyến nghị: MUA"
        elif ml_prediction_value < 30:
            ai_recommendation = f"📉 Giá có dấu hiệu giảm mạnh → Khuyến nghị: BÁN"
        else:
            ai_recommendation = "📊 Thị trường trung lập, theo dõi thêm → Khuyến nghị: GIỮ"
    else:
        ai_recommendation = "Không đủ dữ liệu để dự đoán bằng ML"

    advice = f"""
    - Hỗ trợ hiện tại: {support}<br>
    - Kháng cự hiện tại: {resistance}<br>
    Xu hướng: {trend}. Mua khi giá điều chỉnh về {buy_point}, bán khi chạm {sell_point}.<br>
    <b>Diễn giải chỉ số:</b><br>
    - RSI: Giá trị trên 70 → Quá mua, dưới 30 → Quá bán. Hiện tại: {rsi_14_display}.<br>
    - MACD: Khi đường MACD cắt đường tín hiệu từ dưới lên → Mua, từ trên xuống → Bán. Hiện tại: MACD = {macd_display}, MACD Signal = {macd_signal_display}.<br>
    - MA10, MA20, MA50: MA10 > MA20 → Xu hướng tăng. Hiện tại: MA10 = {ma10_display}, MA20 = {ma20_display}, MA50 = {ma50_display}.<br>
    - Bollinger Bands: Giá chạm Upper → Có thể bán, chạm Lower → Có thể mua. Hiện tại: Upper {bollinger_upper_display}, Lower {bollinger_lower_display}.<br>
    Xác suất tăng giá: {signal_probability * 100:.2f}%.<br>
    Xác suất tăng giá từ ML: {round(float(ml_prediction) * 100, 2)}% (Độ chính xác trung bình: {round(float(ml_accuracy) * 100, 2)}%)<br>
    MAPE của ARIMA: {round(float(arima_mape), 2)}%.<br>
    <b>Phương pháp dự đoán:</b> ARIMA và Random Forest.<br>
    {ai_recommendation}<br>
    Lưu ý: Đặt stop-loss và take-profit để quản lý rủi ro.
    """

    recommendations = f"""
    <h3>📊 Khuyến nghị từ chỉ báo kỹ thuật mở rộng</h3>
    <ul>
        <li>EMA10 ({extra_data['ema10']}) > EMA20 ({extra_data['ema20']}) > EMA50 ({extra_data['ema50']}) → <strong>Xu hướng tăng mạnh</strong>.</li>
        <li>ADX14 = {extra_data['adx14']} → <strong>Xu hướng đủ mạnh để theo dõi giao dịch</strong>.</li>
        <li>Stochastic K = {extra_data['stoch_k']}, D = {extra_data['stoch_d']} → <strong>Gần vùng quá mua</strong>, cân nhắc không mua đuổi.</li>
        <li>VWAP = {extra_data['vwap']}, giá hiện tại > VWAP → <strong>Tín hiệu hỗ trợ MUA</strong>.</li>
        <li>CMF20 = {extra_data['cmf20']} → <strong>Dòng tiền vào thị trường</strong>.</li>
        <li>Volume Oscillator = {extra_data['volume_osc']}% → <strong>Volume tăng mạnh</strong>.</li>
        <li>Ichimoku: Conversion ({extra_data['ichimoku_conversion']}) > Base ({extra_data['ichimoku_base']}) → <strong>Hỗ trợ xu hướng tăng</strong>.</li>
    </ul>
    <p><strong>📌 Tư vấn hành động:</strong> Canh MUA khi giá điều chỉnh nhẹ. Nếu đã mua, có thể GIỮ vị thế. Cẩn trọng khi Stochastic gần quá mua.</p>
    """


    detailed_conclusion = generate_detailed_conclusion(df_clean, last_row, current_price)

    # Sau khi df_clean = calculate_extra_indicators(df_clean)
    
    return {
        'trend': trend,
        'current_price': current_price,
        'rsi_14': rsi_14_display,
        'ma20': ma20_display,
        'ma10': ma10_display,
        'ma50': ma50_display,
        'macd': macd_display,
        'macd_signal': macd_signal_display,
        'atr_14': atr_14,
        'bollinger_upper': bollinger_upper_display,
        'bollinger_lower': bollinger_lower_display,
        'buy_point': buy_point,
        'sell_point': sell_point,
        'forecast_df': forecast_df,
        'candle_pattern': candle_pattern,
        'signal_probability': signal_probability,
        'ml_prediction': ml_prediction,
        'ml_accuracy': ml_accuracy,
        'arima_mape': arima_mape,
        'advice': advice,
        'detailed_conclusion': detailed_conclusion,
        'recommendations': recommendations,
        **extra_data
    }
    

# Route để hiển thị giao diện upload
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route để xử lý file CSV và hiển thị kết quả
@app.post("/analyze", response_class=HTMLResponse)
async def analyze_multiple_files(request: Request, files: List[UploadFile] = File(...)):
    results = []
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    MAX_FILES = 10

    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Tối đa {MAX_FILES} file mỗi lần upload")

    for file in files:
        try:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                results.append({"filename": file.filename, "error": "File quá lớn, tối đa 5MB"})
                continue

            df = pd.read_csv(io.StringIO(content.decode()))

            required_columns = ['time', 'close']
            if not all(col in df.columns for col in required_columns):
                results.append({"filename": file.filename, "error": "Thiếu cột time và close"})
                continue

            if len(df) < 50:
                results.append({"filename": file.filename, "error": "File cần tối thiểu 50 dòng dữ liệu"})
                continue

            if 'index' in df.columns:
                df = df.drop(columns=['index'])

            analysis = analyze_trend(df)
            if analysis is None:
                results.append({"filename": file.filename, "error": "Không đủ dữ liệu để phân tích"})
            else:
                if analysis.get('forecast_df') is not None and isinstance(analysis['forecast_df'], pd.DataFrame):
                    analysis['forecast_df'] = analysis['forecast_df'].astype({'Date': str}).to_dict(orient='records')
                else:
                    analysis['forecast_df'] = []

                results.append({
                    "filename": file.filename,
                    "data": analysis
                })

        except Exception as e:
            error_message = f"Lỗi xử lý file {file.filename}: {str(e)}"
            logging.error(error_message)
            results.append({"filename": file.filename, "error": error_message})

    return templates.TemplateResponse("multi_result.html", {"request": request, "results": results})

@app.on_event("shutdown")
def cleanup():
    if os.path.exists("temp.csv"):
        os.remove("temp.csv")


from fastapi import Form
from vnstock import stock_historical_data

@app.post("/analyze_tickers", response_class=HTMLResponse)
async def analyze_multiple_tickers(
    request: Request,
    symbols: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...)
):
    try:
        symbols_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if not symbols_list:
            raise HTTPException(status_code=400, detail="Chưa nhập mã cổ phiếu hợp lệ.")

        all_results = []
        for symbol in symbols_list:
            try:
                df = stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if len(df) < 50:
                    result = {"symbol": symbol, "error": "Dữ liệu không đủ để phân tích"}
                else:
                    analysis = analyze_trend(df)
                    if analysis:
                        analysis['forecast_df'] = (
                            analysis['forecast_df'].astype({'Date': str}).to_dict(orient='records')
                            if isinstance(analysis['forecast_df'], pd.DataFrame)
                            else []
                        )
                        result = {"symbol": symbol, "data": analysis}
                    else:
                        result = {"symbol": symbol, "error": "Không đủ dữ liệu để phân tích"}
            except Exception as e:
                result = {"symbol": symbol, "error": f"Lỗi: {str(e)}"}
            all_results.append(result)

        return templates.TemplateResponse("multi_ticker_result.html", {
            "request": request,
            "results": all_results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tổng thể: {str(e)}")
