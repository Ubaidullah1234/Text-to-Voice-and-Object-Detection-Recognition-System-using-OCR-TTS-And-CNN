import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const WhatsNewScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>What's New</Text>
      <Text style={styles.content}>
        We are excited to announce the latest updates to our Vision App! 
        In this version, we have improved accessibility features and enhanced user experience. 
        We continue to strive for excellence and welcome your feedback to make our app even better.
      </Text>
      <Text style={styles.trademark}>© 2024 Vision App. All rights reserved.</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F2',
    padding: 20,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    textAlign: 'center',
  },
  content: {
    fontSize: 16,
    marginHorizontal: 10,
  },
  trademark: {
    fontSize: 12,
    marginTop: 20,
    textAlign: 'center',
    color: '#A4A4A4',
  },
});

export default WhatsNewScreen;
