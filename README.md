    🔹 1. Pixel CNN (Digit Recognizer)

- Model CNN aplicat pe date de tip pixeli (vectori numerici)
- Nu folosește imagini .jpg, ci input-uri deja transformate în valori numerice
- Similar cu problema clasică de recunoaștere a cifrelor (ex: MNIST)

Caracteristici:

Input: vectori de pixeli
Model: CNN simplu
Task: clasificare cifre







    🔹 2. Image CNN (Real vs AI Art)
  
- Model CNN aplicat pe imagini reale (.jpg)
- Clasifică imaginile în:
    0 → artă reală
    1 → artă generată de AI

Caracteristici:

Input: imagini RGB
Preprocesare: resize, augmentări
Model: CNN cu mai multe straturi convoluționale
Metrică: F1-score

⚙️ Tehnologii folosite
Python
PyTorch
Pandas
NumPy
PIL



  🚀 Cum rulezi
  Instalează dependințele:
     - pip install torch torchvision pandas pillow scikit-learn
  Rulează scriptul dorit:
     - python source.py
     
  🧠 Scop
    Scopul acestui repository este de a explora:
      - diferențele dintre input-uri numerice vs imagini
      - utilizarea CNN-urilor în contexte diferite

  OBSERVATII:

      --- Codul este scris pentru învățare și experimente, nu este optimizat complet pentru producție ---
