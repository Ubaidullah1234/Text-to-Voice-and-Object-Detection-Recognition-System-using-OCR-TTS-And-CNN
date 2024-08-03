import React, { useEffect, useRef } from 'react';
import { Animated, View, Image, StyleSheet } from 'react-native';
import Logo from '../assets/logo.png';

const SplashScreen = () => {
  const scaleAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(
      scaleAnim,
      {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }
    ).start();
  }, [scaleAnim]);

  return (
    <View style={styles.container}>
      <Animated.Image
        source={Logo}
        style={[styles.logo, { transform: [{ scale: scaleAnim }] }]}
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white', // Set the background color to match your design
  },
  logo: {
    width: '50%', // Adjusted width
    height: 120, // Adjusted height
    resizeMode: 'contain',
  },
});

export default SplashScreen;
