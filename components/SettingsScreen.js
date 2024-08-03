import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Platform } from 'react-native';
import { MaterialIcons, Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

const SettingsScreen = ({ navigation, route }) => {
  const [darkModeEnabled, setDarkModeEnabled] = useState(false);

  const toggleDarkMode = () => {
    setDarkModeEnabled(!darkModeEnabled);
    // Additional logic to handle toggling dark mode
  };

  return (
    <View style={[styles.container, darkModeEnabled && styles.darkContainer]}>
      {/* Support */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkModeEnabled && styles.darkTitle]}>Support</Text>
        <TouchableOpacity
          style={[styles.optionContainer, darkModeEnabled && styles.darkOptionContainer]}
          onPress={() => navigation.navigate('Feedback')}
        >
          <MaterialIcons name="feedback" size={24} color="black" style={styles.optionIcon} />
          <Text style={[styles.optionText, darkModeEnabled && styles.darkOptionText]}>Give Feedback</Text>
          <MaterialIcons name="keyboard-arrow-right" size={24} color="black" style={styles.arrowIcon} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.optionContainer, darkModeEnabled && styles.darkOptionContainer]}
          onPress={() => navigation.navigate('SpeechSettings')}
        >
          <MaterialIcons name="fast-forward" size={24} color="black" style={styles.optionIcon} />
          <Text style={[styles.optionText, darkModeEnabled && styles.darkOptionText]}>Speech</Text>
          <MaterialIcons name="keyboard-arrow-right" size={24} color="black" style={styles.arrowIcon} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.optionContainer, darkModeEnabled && styles.darkOptionContainer]}
          onPress={() => navigation.navigate('Tutorial')}
        >
          <MaterialIcons name="school" size={24} color="black" style={styles.optionIcon} />
          <Text style={[styles.optionText, darkModeEnabled && styles.darkOptionText]}>Tutorial</Text>
          <MaterialIcons name="keyboard-arrow-right" size={24} color="black" style={styles.arrowIcon} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.optionContainer, darkModeEnabled && styles.darkOptionContainer]}
          onPress={() => navigation.navigate('ObjectDetection')}
        >
          <MaterialCommunityIcons name="shape-outline" size={24} color="black" style={styles.optionIcon} />
          <Text style={[styles.optionText, darkModeEnabled && styles.darkOptionText]}>Object Detection</Text>
          <MaterialIcons name="keyboard-arrow-right" size={24} color="black" style={styles.arrowIcon} />
        </TouchableOpacity>
      </View>

      {/* Dark Mode */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkModeEnabled && styles.darkTitle]}>Appearance</Text>
        <View style={[styles.optionContainer, darkModeEnabled && styles.darkOptionContainer]}>
          <Ionicons name={darkModeEnabled ? "moon" : "sunny"} size={24} color="black" style={styles.optionIcon} />
          <Text style={[styles.optionText, darkModeEnabled && styles.darkOptionText]}>Dark Mode</Text>
          <TouchableOpacity
            style={[styles.darkModeToggleButton, darkModeEnabled && styles.darkModeToggleButtonActive]}
            onPress={toggleDarkMode}
          >
            <View style={styles.toggleButtonCircle} />
          </TouchableOpacity>
        </View>
      </View>

      {/* About Us */}
      <View style={styles.section}>
        <Text style={[styles.sectionTitle, darkModeEnabled && styles.darkTitle]}>About Us</Text>
        <TouchableOpacity
          style={[styles.optionContainer, darkModeEnabled && styles.darkOptionContainer]}
          onPress={() => navigation.navigate('WhatsNew')}
        >
          <MaterialIcons name="new-releases" size={24} color="black" style={styles.optionIcon} />
          <Text style={[styles.optionText, darkModeEnabled && styles.darkOptionText]}>What's New</Text>
          <MaterialIcons name="keyboard-arrow-right" size={24} color="black" style={styles.arrowIcon} />
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.optionContainer, darkModeEnabled && styles.darkOptionContainer]}
          onPress={() => navigation.navigate('AboutVision')}
        >
          <MaterialIcons name="visibility" size={24} color="black" style={styles.optionIcon} />
          <Text style={[styles.optionText, darkModeEnabled && styles.darkOptionText]}>About Vision</Text>
          <MaterialIcons name="keyboard-arrow-right" size={24} color="black" style={styles.arrowIcon} />
        </TouchableOpacity>
      </View>

      {/* Bottom Bar */}
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
    backgroundColor: '#F2F2F2',
    paddingTop: Platform.OS === 'ios' ? 20 : 0, // Adjust for iOS status bar
  },
  darkContainer: {
    backgroundColor: '#121212',
  },
  section: {
    marginTop: 5,
  },
  sectionTitle: {
    fontSize: 20,
    color: '#474747',
    fontWeight: 'bold',
    marginLeft: 10,
    marginBottom: 10,
  },
  darkTitle: {
    color: '#FFFFFF',
  },
  optionContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 50,
    backgroundColor: 'white',
    borderRadius: 8,
    marginHorizontal: 10,
    marginBottom: 10,
    paddingHorizontal: 10,
  },
  darkOptionContainer: {
    backgroundColor: '#1E1E1E',
  },
  optionText: {
    color: 'black',
    fontSize: 18,
    marginRight: 'auto',
  },
  darkOptionText: {
    color: '#FFFFFF',
  },
  optionIcon: {
    marginRight: 10,
    color: '#0032A8',
  },
  arrowIcon: {
    marginLeft: 'auto',
    color: '#A4A4A4',
  },
  darkModeToggleButton: {
    marginLeft: 'auto',
    justifyContent: 'center',
    alignItems: 'center',
    width: 50,
    height: 30,
    borderRadius: 15,
    backgroundColor: '#dcdcdc',
  },
  darkModeToggleButtonActive: {
    backgroundColor: '#50D22F',
  },
  toggleButtonCircle: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#fff',
  },
  bottomBar: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    height: 75,
    borderTopColor: '#c9c9c9',
    borderTopWidth: 1,
    backgroundColor: 'white',
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
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
  },
});

export default SettingsScreen;
