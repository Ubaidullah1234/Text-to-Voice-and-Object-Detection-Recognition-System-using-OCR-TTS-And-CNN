import { View, Text, StyleSheet, TouchableOpacity, Platform } from 'react-native';

const FeedbackScreen = ({ navigation }) => {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Please pick an appropriate category for your query.</Text>
      <TouchableOpacity
        style={styles.optionButton}
        onPress={() => navigation.navigate('BugReport')}
      >
        <Text style={styles.optionText}>Bug Report</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.optionButton}
        onPress={() => navigation.navigate('FeatureRequest')}
      >
        <Text style={styles.optionText}>Feature Request</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={styles.optionButton}
        onPress={() => navigation.navigate('OtherQueries')}
      >
        <Text style={styles.optionText}>Other Queries</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F2F2F2',
    marginTop: Platform.OS === 'ios' ? -450 : -200, // Adjust marginTop based on platform
  },
  title: {
    fontSize: 17,
    marginBottom: 20,
    textAlign: 'center',
  },
  optionButton: {
    backgroundColor: '#3167E6',
    borderRadius: 15,
    paddingVertical: 20,
    marginVertical: 5,
    alignItems: 'center',
    width: '80%',
    ...Platform.select({
      android: {
        marginBottom: -50,
        marginTop: 60, // Adjust marginTop for Android
      },
    }),
  },
  optionText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 20,
  },
});

export default FeedbackScreen;