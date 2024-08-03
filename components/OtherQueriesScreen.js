import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, TouchableOpacity, Keyboard, Platform, Alert } from 'react-native';
import { db } from '../FirebaseConfig/FirebaseConfig';
import { collection, addDoc, doc, serverTimestamp } from 'firebase/firestore';

const OtherQueriesScreen = () => {
  const [query, setQuery] = useState('');

  const handleSend = async () => {
    try {
      const userQueriesCollection = collection(db, 'UserQueries'); // collection reference
      await addDoc(userQueriesCollection, {
        query: query,
        createdAt: serverTimestamp(), // using serverTimestamp directly
      });
      console.log('Query sent:', query);
      Alert.alert('Success', 'Your query has been sent!');
      setQuery(''); // Clear the text area after sending
    } catch (error) {
      console.error('Error sending query: ', error);
      Alert.alert('Error', 'There was an issue sending your query.');
    }
  };

  const handleContainerPress = () => {
    Keyboard.dismiss(); // Dismiss the keyboard when tapping outside the text input
  };

  return (
    <TouchableOpacity
      style={styles.container}
      activeOpacity={1} // Prevent TouchableOpacity from fading when pressed
      onPress={handleContainerPress}
    >
      <Text style={styles.title}>You chose:</Text>
      <View style={styles.optionButton}>
        <Text style={styles.optionText}>Other Queries</Text>
      </View>
      <Text style={styles.subtitle}>Please let us know your query or feedback:</Text>
      <TextInput
        style={styles.textInput}
        multiline
        placeholder="Write your query here..."
        value={query}
        onChangeText={text => setQuery(text)}
      />
      <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
        <Text style={styles.sendButtonText}>Send</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#FFFFFF',
    paddingHorizontal: 20,
    marginTop: Platform.OS === 'ios' ? -380 : -60, // Adjust marginTop based on platform
  },
  title: {
    fontSize: 18,
    marginBottom: 10,
    marginLeft: -240,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 20,
    marginLeft: -20,
    marginTop: 30,
  },
  textInput: {
    width: '95%',
    height: 150,
    borderWidth: 1,
    borderColor: 'gray',
    borderRadius: 10,
    padding: 10,
    marginBottom: 20,
    ...Platform.select({
      android: {
        maxHeight: 150, // Adjust maxHeight for Android to prevent text input from expanding indefinitely
      },
    }),
  },
  sendButton: {
    backgroundColor: '#3167E6',
    borderRadius: 15,
    paddingVertical: 15,
    paddingHorizontal: 130,
    marginBottom: 20,
    ...Platform.select({
      android: {
        marginBottom: 70, // Adjust marginBottom for Android
      },
    }),
  },
  sendButtonText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 22,
    textAlign: 'center',
  },
  optionButton: {
    backgroundColor: '#50D22F', // Color the rectangle green
    borderRadius: 15,
    paddingVertical: 15,
    paddingHorizontal: 120,
    marginBottom: 10,
    ...Platform.select({
      android: {
        paddingHorizontal: 90,
        alignItems: 'center', // Adjust marginBottom for Android
      },
    }),
  },
  optionText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 19,
    marginLeft: 10,
  },
});

export default OtherQueriesScreen;
