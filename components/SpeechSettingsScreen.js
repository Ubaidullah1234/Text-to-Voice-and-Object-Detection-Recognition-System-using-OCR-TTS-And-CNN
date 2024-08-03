import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Platform } from 'react-native';
import { FontAwesome5 } from '@expo/vector-icons';
import { MaterialCommunityIcons } from '@expo/vector-icons';

const SpeechSettingsScreen = ({ navigation }) => {
  const [speed, setSpeed] = React.useState(50);

  const decreaseSpeed = () => {
    setSpeed(speed - 10 >= 10 ? speed - 10 : 10);
  };

  const increaseSpeed = () => {
    setSpeed(speed + 10 <= 100 ? speed + 10 : 100);
  };

  const renderBar = () => {
    const greenWidth = (speed - 10) * 2;
    const innerBarWidth = Math.max(greenWidth, 0); // Ensure the width is not negative
    return (
      <View style={styles.baseBar}>
        <View style={styles.innerBar}>
          <View style={[styles.greenBar, { width: `${innerBarWidth}%` }]} />
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.settingsContainer}>
        <View style={styles.descriptionContainer}>
          <Text style={styles.description}>These settings are for the speed of the voice output.</Text>
        </View>
        <Text style={styles.heading}>Speaking Rate</Text>
      </View>
      <View style={styles.speedBar}>
        <TouchableOpacity onPress={decreaseSpeed}>
          <MaterialCommunityIcons name="turtle" size={37} color="black" />
        </TouchableOpacity>
        {renderBar()}
        <TouchableOpacity onPress={increaseSpeed}>
          <MaterialCommunityIcons name="rabbit" size={37} color="black" />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
  },
  settingsContainer: {
    alignItems: 'left',
    marginTop: Platform.OS === 'ios' ? -580 : -360, // Adjusted for both platforms
    marginBottom: 10,
  },
  descriptionContainer: {
    width: '80%',
    marginBottom: 10,
  },
  description: {
    fontSize: 17,
    textAlign: 'left',
  },
  heading: {
    fontSize: 17,
    marginTop: 10,
    fontWeight: 'bold',
    color: '#575757',
  },
  speedBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    width: '90%',
    height: '9%',
    backgroundColor: '#f2f2f2',
    borderRadius: 10,
    marginTop: 15,
    paddingHorizontal: 0,
  },
  baseBar: {
    flex: 1,
    height: 20,
    backgroundColor: '#ccc',
    borderRadius: 10,
    overflow: 'hidden', // Clip the inner bar overflow
  },
  innerBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
    height: '100%',
  },
  greenBar: {
    height: '100%',
    backgroundColor: 'green',
    borderRadius: 10,
  },
  speedIcon: {
    marginHorizontal: 10,
  },
});

export default SpeechSettingsScreen;
