"""
Tracking specific whale wallets
"""
import time
import json
import sys
sys.path.append('/media/henning/Volume/Programming/projectX/src/')
from Helper.helperfunctions import create_table, store_data
from WhaleTracking.whale_tracking import WhaleTracking


if __name__ == "__main__":
    print("los gehts")
    whale_walett_tracking = WhaleTracking()

    balance = whale_walett_tracking.fetch_balances("bc1qskwmt97z9g8vhpvpval7jhs9w6rmk99t5v8yw2")

    print(balance)

    print("ende")