import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Image, TouchableOpacity } from 'react-native';
import SplashScreen from './components/SplashScreen';
import { MaterialIcons, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import Constants from 'expo-constants';
import CameraComponent from './components/CameraComponent';
import SettingsScreen from './components/SettingsScreen';
import Logo from './assets/logo.png';
import CUILogo from './assets/cuilogo.png'; // Import the cuilogo image
import FeedbackScreen from './components/FeedbackScreen';
import SpeechSettingsScreen from './components/SpeechSettingsScreen';
import AboutVisionScreen from './components/AboutVisionScreen';
import TutorialScreen from './components/TutorialScreen';
import WhatsNewScreen from './components/WhatsNewScreen';
import BugReportScreen from './components/BugReportScreen';
import FeatureRequestScreen from './components/FeatureRequestScreen';
import OtherQueriesScreen from './components/OtherQueriesScreen';
import ObjectDetection from './components/ObjectDetection'; // Import the new component

import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator, TransitionPresets } from '@react-navigation/stack';

const Stack = createStackNavigator();

export default function App() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setLoading(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <NavigationContainer>
      {loading ? (
        <SplashScreen />
      ) : (
        <Stack.Navigator
          screenOptions={{
            headerTintColor: 'black', // Set header text color to black
            headerStyle: { backgroundColor: '#ffffff' }, // Set header background color to white
            ...TransitionPresets.SlideFromRightIOS, // Add slide from right transition
          }}
        >
          <Stack.Screen
            name="Home"
            component={HomeScreen}
            options={{ headerShown: false }} // Hide the header on Home screen
          />
          <Stack.Screen name="Settings" component={SettingsScreen} />
          <Stack.Screen name="Feedback" component={FeedbackScreen} />
          <Stack.Screen name="SpeechSettings" component={SpeechSettingsScreen} />
          <Stack.Screen name="AboutVision" component={AboutVisionScreen} />
          <Stack.Screen name="Tutorial" component={TutorialScreen} />
          <Stack.Screen name="WhatsNew" component={WhatsNewScreen} />
          <Stack.Screen name="BugReport" component={BugReportScreen} />
          <Stack.Screen name="FeatureRequest" component={FeatureRequestScreen} />
          <Stack.Screen name="OtherQueries" component={OtherQueriesScreen} />
          <Stack.Screen name="ObjectDetection" component={ObjectDetection} />
        </Stack.Navigator>
      )}
    </NavigationContainer>
  );
}

function HomeScreen({ navigation, route }) {
  return (
    <View style={styles.container}>
      <View style={styles.topBar}>
        <Image source={Logo} style={styles.logo} />
        <Image source={CUILogo} style={styles.cuilogo} />
      </View>

      <CameraComponent />

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
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'space-between',
    backgroundColor: '#f5f5f5', // Changed background color to off white
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between', // Adjust to space out the logos
    paddingHorizontal: 20,
    backgroundColor: '#ffffff', // Set background color to white
    paddingTop: Constants.statusBarHeight + 10, // Adjust for status bar and some padding
    paddingBottom: 10, // Padding bottom to remove white rectangle
    shadowColor: '#000', // Added shadow to the top bar
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 3,
    elevation: 5,
  },
  logo: {
    width: '25%', // Adjusted width to make the logo smaller
    height: 80, // Adjusted height to make the logo smaller
    resizeMode: 'contain',
  },
  cuilogo: {
    width: '20%', // Adjust the width as needed
    height: 60, // Adjust the height as needed
    resizeMode: 'contain',
  },
  bottomBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    height: 75,
    borderTopColor: '#c9c9c9',
    borderTopWidth: 1,
    backgroundColor: 'white',
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
