import cv2
import numpy as np
from sklearn.cluster import KMeans

def rgb_to_cmyk(rgb):
    """Convierte RGB a CMYK con mayor precisión"""
    r, g, b = rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0
    
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, 100  # Negro puro
    
    k = 1 - max(r, g, b)
    c = (1 - r - k) / (1 - k) if (1 - k) != 0 else 0
    m = (1 - g - k) / (1 - k) if (1 - k) != 0 else 0
    y = (1 - b - k) / (1 - k) if (1 - k) != 0 else 0
    
    return tuple(round(x * 100) for x in (c, m, y, k))

def get_dominant_colors(frame, k):
    """Obtiene los k colores dominantes usando K-means"""
    # Redimensionar para mayor eficiencia
    resized = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA)
    
    # Convertir a lista de píxeles y aplicar K-means
    pixels = resized.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    
    # Obtener colores y porcentajes
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = counts / counts.sum()
    
    # Ordenar por porcentaje (de mayor a menor)
    sorted_indices = np.argsort(percentages)[::-1]
    sorted_colors = colors[sorted_indices]
    sorted_percentages = percentages[sorted_indices]
    
    return sorted_colors, sorted_percentages

def draw_color_info(frame, colors, percentages):
    """Dibuja flechas y valores CMYK para cada color dominante"""
    h, w = frame.shape[:2]
    info_panel_height = min(800, 100 * len(colors) + 20)  # Altura máxima de 800px
    info_panel = np.zeros((info_panel_height, 300, 3), dtype=np.uint8) + 220  # Panel gris claro
    
    # Asegurarse de que el frame y el panel tengan la misma altura
    if h != info_panel_height:
        frame = cv2.resize(frame, (w, info_panel_height))
    
    for i, (color, percent) in enumerate(zip(colors, percentages)):
        b, g, r = color
        c, m, y, k = rgb_to_cmyk((r, g, b))
        
        # Posición para la flecha (aleatoria pero consistente)
        arrow_x = np.random.randint(w//4, 3*w//4)
        arrow_y = np.random.randint(h//4, 3*h//4)
        
        # Dibujar flecha apuntando al área de color
        cv2.arrowedLine(frame, (arrow_x, arrow_y), 
                       (w-50, 50 + i*80), (0, 0, 0), 2, tipLength=0.02)
        
        # Dibujar muestra de color en el panel (más compacto)
        cv2.rectangle(info_panel, (10, 10 + i*80), (50, 50 + i*80), 
                     (int(b), int(g), int(r)), -1)
        cv2.rectangle(info_panel, (10, 10 + i*80), (50, 50 + i*80), 
                     (0, 0, 0), 1)
        
        # Mostrar información CMYK (más compacta)
        text = f"C:{c:2d}% M:{m:2d}% Y:{y:2d}% K:{k:2d}%"
        cv2.putText(info_panel, text, (60, 30 + i*80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Mostrar porcentaje de presencia
        cv2.putText(info_panel, f"{percent:.1%}", (60, 55 + i*80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return np.hstack((frame, info_panel))

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Color Detection', cv2.WINDOW_NORMAL)
    
    # Configuración de suavizado para valores dinámicos pero estables
    color_history = []
    smoothing_factor = 5  # Número de frames para promediar
    
    # Configuración del número de colores a detectar
    num_colors = 5  # Puedes cambiar este valor por el número que necesites
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Efecto espejo
        
        # Obtener colores dominantes
        colors, percentages = get_dominant_colors(frame, k=num_colors)
        
        # Suavizar los colores para evitar parpadeo
        color_history.append(colors)
        if len(color_history) > smoothing_factor:
            color_history.pop(0)
        
        # Promediar los colores
        avg_colors = np.mean(color_history, axis=0).astype(int)
        
        # Mostrar información en la pantalla
        display_frame = draw_color_info(frame, avg_colors, percentages)
        
        cv2.imshow('Color Detection', display_frame)
        
        # Teclas para ajustar el número de colores
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('+') and num_colors < 10:  # Límite máximo de 10 colores
            num_colors += 1
            color_history = []  # Reiniciar el historial
        elif key == ord('-') and num_colors > 1:  # Mínimo 1 color
            num_colors -= 1
            color_history = []  # Reiniciar el historial
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
