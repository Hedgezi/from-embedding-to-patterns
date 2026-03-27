import pandas as pd

def extract_text(text_field):
    # В Telegram JSON 'text' может быть строкой или списком объектов
    if isinstance(text_field, str):
        return text_field
    elif isinstance(text_field, list):
        # Собираем текст из кусочков, игнорируя только текст внутри entities
        return "".join([part if isinstance(part, str) else part['text'] for part in text_field])

    return ""

def preprocess_df(df: pd.DataFrame):
    df = df.loc[df["type"].eq("message"), ["date", "text", "from_id"]].copy()

    df["clean_text"] = df["text"].map(extract_text)

    s = df["clean_text"].astype("string")
    mask = s.str.strip().ne("") & s.str.len().gt(5)
    df = df.loc[mask].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print(f"Загружено сообщений для анализа: {len(df)}")

    return df