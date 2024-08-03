import { StatusBar } from "expo-status-bar";
import React, { useState } from "react";
import { Button, StyleSheet, Text, SafeAreaView, ActivityIndicator, View } from "react-native";
import { MaterialIcons } from '@expo/vector-icons'; // Import MaterialIcons from Expo icons library
import * as ImagePicker from "expo-image-picker";
import * as Speech from 'expo-speech'; // Import Text-to-Speech module

export default function App() {
    const [extractedText, setExtractedText] = useState("");
    const [loading, setLoading] = useState(false);

    const pickImageGallery = async () => {
        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            base64: true,
            allowsMultipleSelection: false,
        });
        if (!result.cancelled) {
            performOCR(result.assets[0]);
        }
    };

    const pickImageCamera = async () => {
        let result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            base64: true,
            allowsMultipleSelection: false,
        });
        if (!result.cancelled) {
            performOCR(result.assets[0]);
        }
    };

    const performOCR = (file) => {
        setLoading(true);
        let myHeaders = new Headers();
        myHeaders.append("apikey", "FEmvQr5uj99ZUvk3essuYb6P5lLLBS20");
        myHeaders.append("Content-Type", "multipart/form-data");

        let requestOptions = {
            method: "POST",
            headers: myHeaders,
            body: file,
        };

        fetch("https://api.apilayer.com/image_to_text/upload", requestOptions)
            .then((response) => {
                if (!response.ok) {
                    throw new Error("OCR process failed");
                }
                return response.json();
            })
            .then((result) => {
                setExtractedText(result["all_text"]);
                Speech.speak(result["all_text"]);
                setLoading(false);
            })
            .catch((error) => {
                console.log("OCR Error:", error.message);
                setLoading(false);
            });
    };

    return (
        <SafeAreaView style={styles.container}>
            <Text style={styles.heading}>Welcome</Text>
            <Text style={styles.heading2}>Image to Text App</Text>
            <View style={styles.iconRow}>
                <MaterialIcons name="photo-library" size={60} color="green" onPress={pickImageGallery} />
                <MaterialIcons name="photo-camera" size={60} color="blue" onPress={pickImageCamera} />
            </View>
            {loading && <ActivityIndicator size="large" color="#0000ff" />}
            <Text style={styles.text1}>Extracted text:</Text>
            <Text style={styles.text2}>{extractedText}</Text>
            <StatusBar style="auto" />
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#F2F2F2",
        padding: 20,
        marginTop: -15 ,
    },
    heading: {
        fontSize: 28,
        fontWeight: "bold",
        marginBottom: 10,
        color: "green",
        textAlign: "center",
    },
    heading2: {
        fontSize: 22,
        fontWeight: "bold",
        marginBottom: 10,
        color: "black",
        textAlign: "center",
    },
    iconRow: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        marginBottom: 20,
        marginTop: 20,
    },
    text1: {
        fontSize: 16,
        marginBottom: 5,
        color: "black",
        fontWeight: "bold",
        marginTop: 20,
    },
    text2: {
        fontSize: 16,
        marginBottom: 10,
        color: "black",
        textAlign: "center",
    },
});
