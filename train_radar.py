import cv2
import os
import random
import numpy as np
from ultralytics import YOLO

# 1. YOLOv8 modell betöltése (az első futtatáskor letölti a kb. 6MB-os fájlt)
model = YOLO('yolov8n.pt')

# Mappa elérési útja
FOLDER_PATH = "testpics"

def find_track_polygon(image):
    """
    Stabil, horgonyzott dinamikus trapéz zóna, ami az alsó sínpontokra épül
    és követi a legmegbízhatóbb vonalakat, de nem mászik fel a fák közé.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kép zajszűrése és élkeresés
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Region of Interest (ROI) korlátozása az alsó 60%-ra
    roi_vertices = np.array([
        [(0, height), (width, height), (width, int(height*0.4)), (0, int(height*0.4))]
    ])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Vonalak keresése az alsó ROI-ban
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, threshold=50, minLineLength=50, maxLineGap=100)
    
    # 1. Alapértelmezett horgonyok a kép alján (ha nem találunk semmit)
    l_bottom_x = int(width * 0.28)
    r_bottom_x = int(width * 0.72)
    
    left_lines = []
    right_lines = []
    center_x = width / 2
    
    # 2. Megbízható alsó vonalak keresése a horgonyok finomításához
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            
            # Elvetjük a vízszintes és függőleges vonalakat
            if x1 == x2: continue
            angle = np.arctan2(y1 - y2, abs(x2 - x1)) * 180 / np.pi
            if angle < 15 or angle > 85: continue
            
            # Sínvonal egyenlete (y = mx + b)
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            
            # Hol metszi ez a vonal a kép alját?
            bottom_x = (height - b) / m
            
            # Bal sín: a kép bal feléről/közepéről indul
            if m < -0.2 and 0 < bottom_x < center_x + (width * 0.1):
                left_lines.append((bottom_x, line.flatten()))
            # Jobb sín: a kép jobb feléről/közepéről indul
            elif m > 0.2 and center_x - (width * 0.1) < bottom_x < width:
                right_lines.append((bottom_x, line.flatten()))
                
        # Alsó horgonyok frissítése a legmegbízhatóbb alsó vonalak alapján
        if left_lines:
            # A legalacsonyabb Y pontot kereső vonalat választjuk
            l_bottom_x = int(sorted(left_lines, key=lambda x: x[1][1])[0][0])
        if right_lines:
            # A legalacsonyabb Y pontot kereső vonalat választjuk
            r_bottom_x = int(sorted(right_lines, key=lambda x: x[1][1])[0][0])
            
    # 3. Stabil zóna magasság (ne menjen fel az erdőbe)
    top_y = int(height * 0.48)
    
    # 4. Távlati pont finomítása a távolabbi vonaldarabkák alapján (opcionális)
    # Ez a kanyarkövetést adja vissza, de biztonságos keretek között.
    l_top_x = int(width * 0.45)
    r_top_x = int(width * 0.55)
    
    # Ha van megbízható vonal a távolban, finomítjuk a felső pontokat
    # (Ennek a logikáját az egyszerűség kedvéért most hanyagoljuk, 
    # hogy a zóna fix maradjon, és ne mászzon fel a távoli kaotikus pontokra)
    
    # Dinamikus poligon összeállítása az alsó horgonyokkal
    track_polygon = np.array([
        [l_bottom_x, height],
        [r_bottom_x, height],
        [r_top_x, top_y],
        [l_top_x, top_y]
    ], np.int32)
    
    return track_polygon    
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Kép zajszűrése és élkeresés
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    # Csak a kép alsó 60%-át nézzük, a távoli égbolt/fák nem érdekelnek minket
    roi_vertices = np.array([
        [(0, height), (width, height), (width, int(height*0.4)), (0, int(height*0.4))]
    ])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Vonalak keresése. Paraméterek lazítva, hogy több szaggatott vonalat is megtaláljon
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=150)
    
    left_points = []
    right_points = []
    center_x = width / 2
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            
            # Rendezzük a pontokat úgy, hogy (x1, y1) legyen mindig a lejjebb lévő pont a képen (nagyobb Y)
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
                
            # Zavaró elemek kiszűrése (pl. vízszintes talpfák vagy teljesen függőleges oszlopok)
            if x1 == x2: continue
            angle = np.arctan2(y1 - y2, abs(x2 - x1)) * 180 / np.pi
            if angle < 15 or angle > 85: 
                continue
                
            # Megnézzük a vonalszakasz közepét, hogy eldöntsük, melyik sínhez tartozik
            mid_x = (x1 + x2) / 2
            if mid_x < center_x:
                left_points.extend([(x1, y1), (x2, y2)])
            else:
                right_points.extend([(x1, y1), (x2, y2)])
                
    # Segédfüggvény: a pontokat lentről felfelé haladva sima görbévé fűzi
    def get_smooth_path(points):
        if not points: return []
        bins = {}
        # 20 pixeles vízszintes sávokba (Y-ok) csoportosítjuk a pontokat
        for x, y in points:
            bin_y = (y // 20) * 20
            if bin_y not in bins:
                bins[bin_y] = []
            bins[bin_y].append(x)
            
        path = []
        # Lentről felfelé haladunk (Y csökken)
        for bin_y in sorted(bins.keys(), reverse=True):
            # A sávban lévő X koordináták mediánját vesszük, így a kiugró hibás élek (pl kavics) kiesnek
            med_x = int(np.median(bins[bin_y]))
            path.append((med_x, bin_y))
        return path
        
    # Kiszámítjuk a bal és jobb sín ívét (ha van benne kanyar, azt is leköveti)
    left_path = get_smooth_path(left_points)
    right_path = get_smooth_path(right_points)
    
    # Ha van legalább egy-egy vonalunk bal és jobb oldalon
    if len(left_path) > 1 and len(right_path) > 1:
        # Megnézzük, meddig (milyen magasra) látjuk a síneket.
        # Az utolsó elem a legkisebb Y (legmagasabb pont a képernyőn).
        min_y_left = left_path[-1][1]
        min_y_right = right_path[-1][1]
        
        # Azt az Y magasságot választjuk, ameddig még MINDKÉT sín elért (max = lejjebb lévő y)
        stop_y = max(min_y_left, min_y_right) 
        
        # Levágjuk a felesleget mindkét oldalról
        final_left = [(x, y) for x, y in left_path if y >= stop_y]
        final_right = [(x, y) for x, y in right_path if y >= stop_y]
        
        if final_left and final_right:
            # Horgonyozzuk le a képernyő aljához, hogy ne legyen rés
            final_left.insert(0, (final_left[0][0], height))
            final_right.insert(0, (final_right[0][0], height))
            
            # A SZUPER POLIGON:
            # Végigmegyünk a bal sínen lentről fel (final_left),
            # majd összekötjük a jobb sín tetejével, és végigmegyünk a jobb sínen fentről le (final_right[::-1])
            polygon_points = final_left + final_right[::-1]
            return np.array(polygon_points, np.int32)
            
    # --- BIZTONSÁGI TARTALÉK (FALLBACK) ---
    # Ha teljesen elveszítette a fonalat a kép (pl. hófúvás), jön a fix trapéz
    return np.array([
        [int(width * 0.3), height],
        [int(width * 0.7), height],
        [int(width * 0.55), int(height * 0.6)],
        [int(width * 0.45), int(height * 0.6)]
    ], np.int32)

def main():
    if not os.path.exists(FOLDER_PATH):
        print(f"Hiba: A '{FOLDER_PATH}' mappa nem létezik! Kérlek hozd létre és tegyél bele képeket.")
        return

    # Képfájlok kilistázása
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(FOLDER_PATH) if f.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"A '{FOLDER_PATH}' mappa üres! Dobj bele pár képet.")
        return

    print("Program indítása... A kilépéshez nyomd meg a 'q' gombot.")

    while True:
        # 1. Random kép kiválasztása
        img_name = random.choice(images)
        img_path = os.path.join(FOLDER_PATH, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue

        frame = cv2.resize(frame, (1024, 768)) # Átméretezés, hogy kiférjen a képernyőre
        
        # 2. Sínpár (veszélyzóna) megkeresése és poligon generálása
        track_polygon = find_track_polygon(frame)
        
        # 3. Objektumok felismerése a YOLOv8-cal
        results = model(frame, verbose=False)
        
        obstacle_detected = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box (befoglaló doboz) koordináták
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                
                # Miket tekintünk akadálynak? (Ember:0, Autó:2, Motor:3, Busz:5, Vonat:6 (ha szembe jön), Állatok stb.)
                # Ha minden objektumot akadálynak akarsz venni, vedd ki a feltételt.
                
                # Objektum "talpa" (középen lent) -> Ezt vizsgáljuk, hogy a sínen van-e
                bottom_center = (int((x1 + x2) / 2), y2)
                
                # 4. Belelógás ellenőrzése (Benne van-e a pont a poligonban?)
                # A pointPolygonTest > 0 ha benne van, = 0 ha a határán, < 0 ha kint van.
                is_inside = cv2.pointPolygonTest(track_polygon, bottom_center, False) >= 0
                
                if is_inside:
                    obstacle_detected = True
                    # Piros doboz rajzolása az akadály köré
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.circle(frame, bottom_center, 5, (0, 0, 255), -1)
                else:
                    # Zöld doboz, ha nincs a sínen
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 5. Rajzolás a képre
        # Sínpár poligon kirajzolása (félig átlátszó kékkel)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [track_polygon], (255, 0, 0))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.polylines(frame, [track_polygon], isClosed=True, color=(255, 0, 0), thickness=2)

        # 6. Üzenet feldobása, ha van akadály
        if obstacle_detected:
            # Sötét háttér a szövegnek a jobb olvashatóságért
            cv2.rectangle(frame, (50, 50), (950, 150), (0, 0, 0), -1)
            cv2.putText(frame, "FIGYELEM! AKADALY A SINEN!", (70, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)

        # Kép megjelenítése
        cv2.imshow("Vasuti Akadaly Detektor", frame)

        # 7. 1 másodperc (1000 ms) várakozás, vagy kilépés 'q' lenyomására
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()