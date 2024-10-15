from PIL import Image, ImageFilter

def blur_image(image, blur_radius=2):
    """
    对给定的图像应用高斯模糊。
    
    Args:
        image (PIL.Image): 输入的PIL图像。
        blur_radius (float): 模糊半径。
        
    Returns:
        PIL.Image: 模糊处理后的图像。
    """
    return image.filter(ImageFilter.GaussianBlur(blur_radius))
