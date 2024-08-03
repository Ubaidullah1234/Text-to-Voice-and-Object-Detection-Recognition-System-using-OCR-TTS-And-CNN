import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const TutorialScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Tutorial</Text>
      <Text style={styles.tutorialStep}>1. Install the Vision App from the app store.</Text>
      <Text style={styles.tutorialStep}>2. Launch the app and create an account if prompted.</Text>
      <Text style={styles.tutorialStep}>3. Explore the various features such as:</Text>
      <Text style={styles.tutorialSubStep}>- Feedback: Provide feedback to help us improve the app.</Text>
      <Text style={styles.tutorialSubStep}>- Settings: Customize your app settings, including dark mode.</Text>
      <Text style={styles.tutorialSubStep}>- About Us: Learn more about the creators and purpose of the app.</Text>
      <Text style={styles.tutorialSubStep}>- What's New: Stay updated on the latest features and improvements.</Text>
      <Text style={styles.tutorialStep}>4. Enjoy the benefits of the Vision App:</Text>
      <Text style={styles.tutorialSubStep}>- Improved accessibility for visually impaired individuals.</Text>
      <Text style={styles.tutorialSubStep}>- Enhanced user experience with intuitive design.</Text>
      <Text style={styles.tutorialSubStep}>- Empowerment through technology to navigate the digital world.</Text>
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
  tutorialStep: {
    fontSize: 16,
    marginHorizontal: 10,
    marginBottom: 5,
  },
  tutorialSubStep: {
    fontSize: 14,
    marginHorizontal: 20,
    marginBottom: 5,
    color: '#474747',
  },
  trademark: {
    fontSize: 12,
    marginTop: 20,
    textAlign: 'center',
    color: '#A4A4A4',
  },
});

export default TutorialScreen;
 