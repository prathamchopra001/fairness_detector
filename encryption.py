import hashlib
import os
import logging
from cryptography.fernet import Fernet, InvalidToken
from typing import Optional

class SecureImageStorage:
    def __init__(self, storage_path: str):
        """
        Initialize secure image storage with encryption capabilities
        
        Args:
            storage_path (str): Directory for storing encrypted images
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Securely generate and store encryption key
        self.key_path = os.path.join(storage_path, 'encryption_key.key')
        
        if not os.path.exists(self.key_path):
            encryption_key = Fernet.generate_key()
            with open(self.key_path, 'wb') as key_file:
                key_file.write(encryption_key)
        
        with open(self.key_path, 'rb') as key_file:
            self.encryption_key = key_file.read()
        
        self.cipher_suite = Fernet(self.encryption_key)

    def anonymize_filename(self, original_filename: str) -> str:
        """
        Create a cryptographic hash of the filename
        
        Args:
            original_filename (str): Original filename to anonymize
        
        Returns:
            str: Anonymized filename
        """
        return hashlib.sha256(original_filename.encode()).hexdigest()[:16]

    def encrypt_image(self, image_path: str) -> str:
        """
        Encrypt image file and store securely
        
        Args:
            image_path (str): Path to the image file to encrypt
        
        Returns:
            str: Path to the encrypted image file
        """
        try:
            with open(image_path, 'rb') as file:
                file_data = file.read()
            
            encrypted_data = self.cipher_suite.encrypt(file_data)
            
            anonymized_name = self.anonymize_filename(os.path.basename(image_path))
            encrypted_path = os.path.join(self.storage_path, f"{anonymized_name}.encrypted")
            
            with open(encrypted_path, 'wb') as encrypted_file:
                encrypted_file.write(encrypted_data)
            
            return encrypted_path
        
        except Exception as e:
            logging.error(f"Encryption error: {e}")
            raise

    def decrypt_image(self, encrypted_path: str, output_path: Optional[str] = None) -> str:
        """
        Decrypt an encrypted image file
        
        Args:
            encrypted_path (str): Path to the encrypted image file
            output_path (str, optional): Destination path for decrypted image
        
        Returns:
            str: Path to the decrypted image file
        """
        try:
            # Read encrypted data
            with open(encrypted_path, 'rb') as encrypted_file:
                encrypted_data = encrypted_file.read()
            
            # Decrypt the data
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            
            # Determine output path
            if output_path is None:
                filename = os.path.basename(encrypted_path).replace('.encrypted', '')
                output_path = os.path.join(self.storage_path, f"decrypted_{filename}")
            
            # Write decrypted data
            with open(output_path, 'wb') as decrypted_file:
                decrypted_file.write(decrypted_data)
            
            return output_path
        
        except InvalidToken:
            logging.error("Decryption failed: Invalid or corrupted encryption key")
            raise ValueError("Cannot decrypt: Invalid encryption key")
        
        except Exception as e:
            logging.error(f"Decryption error: {e}")
            raise

    def secure_delete(self, file_path: str):
        """
        Securely delete a file by overwriting its contents
        
        Args:
            file_path (str): Path to the file to be securely deleted
        """
        try:
            file_size = os.path.getsize(file_path)
            with open(file_path, 'wb') as file:
                file.write(os.urandom(file_size))
            os.remove(file_path)
        except Exception as e:
            logging.error(f"Secure deletion error: {e}")

# Example Usage
if __name__ == "__main__":
    secure_storage = SecureImageStorage('./secure_images')
    
    # Encrypt an image
    encrypted_image = secure_storage.encrypt_image('original_image.jpg')
    
    # Decrypt the image
    decrypted_image = secure_storage.decrypt_image(encrypted_image)
    
    # Optional: Securely delete original files
    secure_storage.secure_delete('original_image.jpg')