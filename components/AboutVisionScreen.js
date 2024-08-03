import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const AboutUsScreen = () => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>About Us</Text>
      <Text style={styles.content}>
        Our Vision App is dedicated to providing innovative solutions for visual impairment. 
        We believe in leveraging technology to improve the lives of individuals with visual challenges. 
        Our team is committed to creating accessible and user-friendly applications that empower our users.
      </Text>
      <Text style={styles.boldText}>Supervisor: Dr Waqas Jadoon</Text>
      <Text style={styles.boldText}>Group Members: </Text>
      <Text style={styles.boldText}>Nouman Khalid </Text>
      <Text style={styles.boldText}>Ubaidullah Mushtaq </Text>
      <Text style={styles.boldText}>Abdullah Naveed </Text>

      
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
  boldText: {
    fontWeight: 'bold',
    marginTop: 10, // Optional: Add some spacing between these texts and the content above
    marginLeft: 10,
  },
  trademark: {
    fontSize: 12,
    marginTop: 20,
    textAlign: 'center',
    color: '#A4A4A4',
  },
});

export default AboutUsScreen;
