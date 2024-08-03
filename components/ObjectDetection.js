import React, { useState } from 'react';
import { View, Button, Image, Text, StyleSheet, Alert, ActivityIndicator, TouchableOpacity } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import { MaterialIcons, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

const ObjectDetection = ({ navigation, route }) => {
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState('');
    const [loading, setLoading] = useState(false);

    const serverUrl = 'http://192.168.1.2:5000';

    const pickImage = async () => {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert('Permission Denied', 'Camera roll permissions are required to use this feature.');
            return;
        }

        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            quality: 1,
        });

        if (!result.cancelled) {
            setImage(result.assets[0].uri);
        }
    };

    const takePhoto = async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert('Permission Denied', 'Camera permissions are required to use this feature.');
            return;
        }

        let result = await ImagePicker.launchCameraAsync({
            allowsEditing: true,
            quality: 1,
        });

        if (!result.cancelled) {
            setImage(result.assets[0].uri);
        }
    };

    const predictImage = async () => {
        if (!image) {
            Alert.alert('No Image', 'Please select or capture an image first.');
            return;
        }

        setLoading(true);
        setPrediction('');

        let formData = new FormData();
        formData.append('image', {
            uri: image,
            name: 'photo.jpg',
            type: 'image/jpeg'
        });

        try {
            const response = await axios.post(`${serverUrl}/predict`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            console.log(response.data);

            setPrediction(response.data.prediction);

            Speech.speak(response.data.prediction, { language: 'en' });

        } catch (error) {
            console.error('Error predicting:', error);
            Alert.alert('Prediction Error', error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <View style={styles.container}>
            <View style={styles.topButtons}>
                <TouchableOpacity
                    style={{ ...styles.button, backgroundColor: '#2196F3' }}
                    onPress={pickImage}
                >
                    <Text style={styles.buttonText}>Pick an Image from Gallery</Text>
                </TouchableOpacity>
                <View style={{ paddingVertical: 1 }} />
                <TouchableOpacity
                    style={{ ...styles.button, backgroundColor: '#2196F3' }}
                    onPress={takePhoto}
                >
                    <Text style={styles.buttonText}>Take a Photo</Text>
                </TouchableOpacity>
            </View>

            <View style={styles.imageContainer}>
                {image && <Image source={{ uri: image }} style={styles.image} />}
                {image === null && (
                    <View style={styles.imagePlaceholder}>
                        <Text>No Image Selected</Text>
                    </View>
                )}
            </View>

            <TouchableOpacity style={styles.predictButton} onPress={predictImage} disabled={loading}>
                <Text style={styles.predictButtonText}>Predict Image</Text>
            </TouchableOpacity>

            {loading && <ActivityIndicator size="large" color="#0000ff" />}
            {prediction !== '' && <Text style={styles.prediction}>Prediction: {prediction}</Text>}

            <View style={styles.bottomBar}>
                <TouchableOpacity
                    style={styles.bottomBarItem}
                    onPress={() => navigation.navigate('Home')}
                >
                    <MaterialIcons name="home-filled" size={35} color="black" />
                    {route.name === 'Home' && <View style={styles.activeLine} />}
                </TouchableOpacity>
                <TouchableOpacity
                    style={styles.bottomBarItem}
                    onPress={() => navigation.navigate('ObjectDetection')}
                >
                    <MaterialCommunityIcons name="shape-outline" size={35} color="black" />
                    {route.name === 'ObjectDetection' && <View style={styles.activeLine} />}
                </TouchableOpacity>
                <TouchableOpacity
                    style={styles.bottomBarItem}
                    onPress={() => navigation.navigate('Settings')}
                >
                    <Ionicons name="settings" size={35} color="black" />
                    {route.name === 'Settings' && <View style={styles.activeLine} />}
                </TouchableOpacity>
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'top',
        alignItems: 'center',
        backgroundColor: '#f5f5f5',
        marginTop:20,
    },
    topButtons: {
        marginTop: 20,
        marginBottom: 10,
    },
    button: {
        backgroundColor: '#2196F3',
        paddingVertical: 12,
        paddingHorizontal: 24,
        borderRadius: 25,
        marginBottom: 10,
    },
    buttonText: {
        color: 'white',
        fontSize: 18,
        textAlign: 'center',
    },
    imageContainer: {
        borderWidth: 2,
        borderColor: 'black',
        borderRadius: 5,
        width: '70%',
        aspectRatio: 1,
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: 20,
        marginBottom: 20,
    },
    image: {
        width: '100%',
        height: '100%',
        resizeMode: 'contain',
    },
    imagePlaceholder: {
        justifyContent: 'center',
        alignItems: 'center',
        width: '100%',
        height: '100%',
        backgroundColor: '#e0e0e0',
    },
    predictButton: {
        backgroundColor: '#2196F3',
        paddingHorizontal: 20,
        paddingVertical: 10,
        borderRadius: 5,
        marginTop: 20,
    },
    predictButtonText: {
        color: 'white',
        fontSize: 18,
    },
    prediction: {
        marginTop: 20,
        fontSize: 18,
        fontWeight: 'bold',
    },
    bottomBar: {
        flexDirection: 'row',
        justifyContent: 'space-around',
        alignItems: 'center',
        height: 75,
        borderTopColor: '#c9c9c9',
        borderTopWidth: 1,
        backgroundColor: 'white',
        width: '100%',
        position: 'absolute',
        bottom: 0,
    },
    bottomBarItem: {
        paddingVertical: 10,
        alignItems: 'center',
    },
    activeLine: {
        width: '80%',
        height: 3,
        backgroundColor: 'black',
        position: 'absolute',
        bottom: 0,
        marginBottom: 3,
    },
});

export default ObjectDetection;
