import pygame

class Track:
    def __init__(self, image_path):
        self.image = pygame.image.load(image_path).convert_alpha()
        self.road_color = (120,91,100)  # цвет дороги
        self.rect = self.image.get_rect()
        self.mask = (0,0,0)
        
    def check_collision(self, car_mask, car_position):
        """Проверка столкновения машины с границами трассы"""
        offset = (car_position[0] - self.rect.x, car_position[1] - self.rect.y)
        return self.mask.overlap(car_mask, offset)
    
    def create_mask_from_color(self, surface):
        """Создает маску на основе цвета дороги"""
        width, height = surface.get_size()
        mask = pygame.Mask((width, height))
        
        # Проходим по всем пикселям
        for y in range(height):
            for x in range(width):
                pixel_color = surface.get_at((x, y))
                # Если пиксель НЕ цвета дороги - добавляем в маску
                if pixel_color != self.road_color:
                    mask.set_at((x, y), 1)
        self.mask = mask

    
        
