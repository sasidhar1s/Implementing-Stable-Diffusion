from PIL import Image
from torchvision import transforms as T

class ImageProcessor:
    def __init__(self, image_size=512):
        self.image_size = image_size
        
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5], inplace=True)
        ])

    def preprocess_image(self, image_path):
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {image_path}. Skipping. Error: {e}")
            return None

        width, height = image.size

        
        if width < self.image_size or height < self.image_size:
            return None

        if width == height:
            
            resizer = T.Resize((self.image_size, self.image_size), antialias=True)
            image = resizer(image)
        else:
            
            crop_size = min(width, height)
            cropper = T.CenterCrop(crop_size)
            resizer = T.Resize((self.image_size, self.image_size), antialias=True)
            image = resizer(cropper(image))
            
        return self.transform(image)