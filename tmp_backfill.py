import sqlite3

conn = sqlite3.connect('event_registry.db')
conn.execute("UPDATE Master_Event_Log SET Source_Video_Path = 'D:/projects/capstone/dataset/testVideo.mp4' WHERE Event_ID = 'EVT_2395AE3BD1EB'")
conn.commit()

# Verify
row = conn.execute("SELECT Event_ID, Source_Video_Path, Trigger_Time FROM Master_Event_Log WHERE Event_ID = 'EVT_2395AE3BD1EB'").fetchone()
print(f"Event: {row[0]}, Source: {row[1]}, Trigger: {row[2]}")
conn.close()
