Features

- **Text-to-Voice Conversion**: Converts extracted text to speech for visually impaired users.
- **Object Detection**: Identifies and classifies objects in images using a CNN model.
- **OCR Integration**: Extracts text from images to be converted into speech.
- **Firebase Integration**: Manages user interactions through Firebase with the following components:
  - **Ask Queries**: Users can submit queries.
  - **Bug Report**: Users can report bugs encountered in the application.
  - **Feature Request**: Users can request new features or improvements.
 
- React Native
- Expo 
- Python (for running the backend server)
- Firebase account
 **Install frontend dependencies:**

    ```bash
    npm install
    ```

 **Set up Firebase:**
   - Create a Firebase project.
   - Set up Firestore and create collections for queries, bug reports, and feature requests.
   - Update `FirebaseConfig.js` with your Firebase project credentials.

 **Install backend dependencies (if applicable):**

    ```bash
    pip install 
    ```



### Running the React Native App

1. Start the Expo development server:

    ```bash
    npm run start
    ```

2. Use the Expo app on your mobile device.

### Running the Backend Server

1. Ensure you have the correct dependencies installed and start the server:

    ```bash
    python server.py
    ```
