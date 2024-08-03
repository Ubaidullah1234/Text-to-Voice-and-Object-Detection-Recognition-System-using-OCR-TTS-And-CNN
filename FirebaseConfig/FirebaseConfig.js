import {initializeApp} from "firebase/app";
import {getAuth} from "@firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyCLggvkc110_5YPB19W_8YZU_zxeb6CL3w",
  authDomain: "final-year-project-3c607.firebaseapp.com",
  projectId: "final-year-project-3c607",
  storageBucket: "final-year-project-3c607.appspot.com",
  messagingSenderId: "52978931095",
  appId: "1:52978931095:web:d973ffdc3298ea3393e0d2",
  measurementId: "G-ZH2S0WCSZN"
};


const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);