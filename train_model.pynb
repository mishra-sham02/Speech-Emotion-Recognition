def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=2.5, sr=22050, offset=0.5)

        # Initialize feature list
        features = []

        # Extract features if signal length is sufficient
        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            features.append(np.mean(mfccs, axis=1))  # Mean across time

        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_feature = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
            features.append(np.mean(chroma_feature, axis=1))

        if mel:
            mel_feature = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            features.append(np.mean(mel_feature, axis=1))

        return np.hstack(features)  # Combine all features into one vector

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return np.zeros(40)  # Return a zero vector as a fallback

def load_data(test_size=0.2):
    x, y = [], []
    data_path = "/content/audio_speech_actors_01-24"
    for file in glob.glob(os.path.join(data_path, "Actor_*", "*.wav")):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


x_train,x_test,y_train,y_test=load_data(test_size=0.25)
# Verify the split
print((x_train.shape[0], x_test.shape[0]))

print(f'Features extracted: {x_train.shape[1]}')

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

import joblib
joblib.dump(model, "trained_model.pkl")
