import cv2
import numpy as np
import onnxruntime as rt
import json

# Percorso del modello ONNX
MODEL_PATH = "mobilenetv2-7.onnx"  # Nome del file modello
LABELS_PATH = "imagenet-simple-labels.json"  # Percorso del file JSON con le etichette

def load_model():
    """Carica il modello ONNX."""
    print("Caricamento del modello ONNX...")
    session = rt.InferenceSession(MODEL_PATH)
    print("Modello ONNX caricato con successo!")
    return session

def load_labels():
    """Carica le etichette di ImageNet dal file JSON."""
    try:
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)  # Carica le etichette come lista
        print("Etichette caricate con successo!")
        return labels
    except FileNotFoundError:
        print(f"File {LABELS_PATH} non trovato. Usa etichette predefinite.")
        return [f"Classe {i}" for i in range(1000)]

def preprocess_frame(frame):
    """Preprocessa il frame per il modello ONNX."""
    resized = cv2.resize(frame, (224, 224))  # Dimensioni standard MobileNet
    img_array = np.transpose(resized, (2, 0, 1)).astype('float32')  # Cambia ordine dei canali
    img_array = np.expand_dims(img_array, axis=0)  # Aggiungi dimensione batch
    img_array /= 255.0  # Normalizzazione
    return img_array

def predict_objects(session, frame):
    """Esegue predizioni sul frame usando il modello ONNX."""
    input_name = session.get_inputs()[0].name
    processed_frame = preprocess_frame(frame)
    predictions = session.run(None, {input_name: processed_frame})
    return predictions

def draw_predictions(frame, predictions, labels):
    """Disegna le predizioni sul frame."""
    probabilities = predictions[0][0]  # Output MobileNet: probabilità per ogni classe
    top_class_index = np.argmax(probabilities)  # Trova l'indice con la probabilità più alta
    
    if top_class_index < len(labels):  # Controlla che l'indice sia valido
        label = labels[top_class_index]
        confidence = probabilities[top_class_index]
        text = f"{label}: {confidence:.2f}"
    else:
        text = "Classe sconosciuta"

    # Disegna il testo sul frame
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    """Funzione principale per il riconoscimento in tempo reale."""
    try:
        # Carica il modello e le etichette
        model = load_model()
        labels = load_labels()

        # Inizializza la webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Errore nell'aprire la webcam.")
            return

        print("Premi 'q' per uscire.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Errore nel catturare il frame.")
                break

            # Applica l'effetto specchio
            frame = cv2.flip(frame, 1)

            # Esegui predizioni sul frame
            predictions = predict_objects(model, frame)

            # Disegna le predizioni
            draw_predictions(frame, predictions, labels)

            # Mostra il feed video con predizioni
            cv2.imshow("Riconoscimento Oggetti - ONNX", frame)

            # Esci con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Si è verificato un errore: {e}")

    finally:
        # Rilascia risorse
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("Risorse rilasciate correttamente.")

if __name__ == "__main__":
    main()