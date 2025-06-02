import pandas as pd
import re
from datetime import datetime
import emoji

def replace_links(text):
    """
    Replaces all URLs in the input text with the string '[LINK]'.

    Args:
        text (str): The input text.

    Returns:
        str: The text with URLs replaced by '[LINK]'.
    """
    return re.sub(r'http\S+', '[LINK]', text)

def parse_whatsapp_chat(input_file, remove_media=True, convert_emojis=True, max_gap_minutes=60):
    """
    Parses a WhatsApp chat export file and groups messages into conversations based on time gaps.

    Args:
        input_file (str): Path to the WhatsApp chat text file.
        remove_media (bool): If True, removes messages indicating omitted media.
        convert_emojis (bool): If True, converts emojis to their text shortcodes.
        max_gap_minutes (int): Maximum gap (in minutes) between messages to consider them in the same conversation.

    Returns:
        pd.DataFrame: DataFrame with columns ['datetime', 'sender', 'message', 'conversation_id'].
    """
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2}), (\d{1,2}:\d{2}\s[ap]m) - (.*?): (.*)$')
    data = []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            match = pattern.match(line)
            if match:
                date_str, time_str, sender, message = match.groups()
                dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%y %I:%M %p")
                data.append((dt, sender, message))
            else:
                if data:
                    # Append continuation line to the last message
                    data[-1] = (data[-1][0], data[-1][1], data[-1][2] + " " + line)

    df = pd.DataFrame(data, columns=['datetime', 'sender', 'message'])

    if remove_media:
        df = df[~df['message'].str.contains(r'<Media omitted>', na=False)]

    if convert_emojis:
        df['message'] = df['message'].apply(lambda x: emoji.demojize(x))

    df['message'] = df['message'].apply(replace_links)

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # Assign conversation_id based on time gaps
    conversation_id = 0
    conversation_ids = []

    for i in range(len(df)):
        if i == 0:
            conversation_ids.append(conversation_id)
        else:
            time_diff = (df.loc[i, 'datetime'] - df.loc[i - 1, 'datetime']).total_seconds() / 60
            if time_diff > max_gap_minutes:
                conversation_id += 1
            conversation_ids.append(conversation_id)

    df['conversation_id'] = conversation_ids

    return df

def save_parsed_chat(df, output_file):
    """
    Saves the parsed chat DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_file (str): Path to the output CSV file.
    """
    df.to_csv(output_file, index=False)
