import os
import json
import getpass
from cryptography.fernet import Fernet

def setup_secure_credentials():
    print("==========================================")
    print(" Database Secrets Encryption Utility")
    print("==========================================")
    print("This tool will securely encrypt your Postgres credentials.")
    print("The encryption key will be disguised as 'data/model_config.bin'.")
    print("The encrypted secrets will be disguised as 'data/model_weights.enc'.")
    print("==========================================\n")

    # Gather credentials
    host = input("Enter Postgres Host [localhost]: ") or "localhost"
    port = input("Enter Postgres Port [5432]: ") or "5432"
    dbname = input("Enter Database Name [abc]: ") or "abc"
    user = input("Enter Database User [postgres]: ") or "postgres"
    
    # Use getpass to hide the password input
    password = getpass.getpass("Enter Database Password (hidden): ")

    payload = {
        "DB_HOST": host,
        "DB_PORT": port,
        "DB_NAME": dbname,
        "DB_USER": user,
        "DB_PASS": password
    }

    # Generate a new secure Master Key
    print("\nGenerating secure AES encryption key...")
    master_key = Fernet.generate_key()
    fernet = Fernet(master_key)

    # Encrypt the JSON payload
    print("Encrypting payload...")
    encrypted_payload = fernet.encrypt(json.dumps(payload).encode('utf-8'))

    # Determine paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    key_path = os.path.join(data_dir, "model_config.bin")
    secrets_path = os.path.join(data_dir, "model_weights.enc")

    # Write the key to disk (disguised)
    with open(key_path, "wb") as key_file:
        key_file.write(master_key)
        
    # Write the encrypted secrets to disk (disguised)
    with open(secrets_path, "wb") as secrets_file:
        secrets_file.write(encrypted_payload)

    print("\nSuccess! Credentials encrypted safely.")
    print(f"Key saved to: {key_path}")
    print(f"Encrypted secrets saved to: {secrets_path}")
    print("\nIMPORTANT: On your Linux server, run the following to restrict access to the key:")
    print(f"chmod 400 {key_path}")

if __name__ == "__main__":
    setup_secure_credentials()
