import hashlib
import base58

def address_to_scripthash(address:str) -> str:
    """Convert a Bitcoin address to an Electrum scripthash."""
    decoded = base58.b58decode_check(address)
    pubkey_hash = decoded[1:]
    scripthash = hashlib.sha256(pubkey_hash).digest()[::-1].hex()
    return scripthash