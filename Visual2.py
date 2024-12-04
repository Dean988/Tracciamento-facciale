import cv2
import dlib

# Carica il rilevatore di volti di dlib
detector = dlib.get_frontal_face_detector()
# Carica il predittore di punti di riferimento facciali
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Inizializza il tracker di correlazione di dlib
tracker = dlib.correlation_tracker()

# Apri la webcam
cap = cv2.VideoCapture(0)

tracking_face = False
recognition_enabled = True  # Flag per abilitare/disabilitare il riconoscimento

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converti l'immagine in scala di grigi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if recognition_enabled:
        if not tracking_face:
            # Rileva i volti nel frame
            faces = detector(gray)
            if len(faces) > 0:
                # Usa il primo volto rilevato per iniziare il tracciamento
                face = faces[0]
                tracker.start_track(frame, face)
                tracking_face = True
        else:
            # Aggiorna il tracciamento
            tracking_quality = tracker.update(frame)
            if tracking_quality >= 8.75:
                pos = tracker.get_position()
                x = int(pos.left())
                y = int(pos.top())
                w = int(pos.width())
                h = int(pos.height())
                # Disegna il rettangolo intorno al volto tracciato
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Trova i punti di riferimento del volto
                rect = dlib.rectangle(x, y, x + w, y + h)
                landmarks = predictor(gray, rect)
                # Disegna i punti di riferimento
                for n in range(68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            else:
                tracking_face = False

    # Mostra il frame
    cv2.imshow("Riconoscimento e Tracciamento del Volto", frame)

    # Gestisci input da tastiera
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('z'):
        recognition_enabled = not recognition_enabled
        tracking_face = False  # Resetta il tracciamento quando si disabilita

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()