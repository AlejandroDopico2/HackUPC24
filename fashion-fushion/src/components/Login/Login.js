import React, { useState } from 'react';
import { supabase } from '../../utils/supabase';
import { useNavigate } from 'react-router-dom';

const SignIn = () => {
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();

  const handleEmailSignIn = async (event) => {
    event.preventDefault();
    const { data, error } = await supabase.auth.signInWithPassword({
      email: event.target.email.value,
      password: event.target.password.value,
    });

    if (error) {
      setErrorMessage(error.message);
      console.error('Error signing up:', error.message);
    } else {
      console.log('User signed in:', data.user);
      navigate('/upload');
    }
  };

  return (
    <div className="form-container">
      <div className="form-form">
        <h2>Sign in</h2>
        <form onSubmit={handleEmailSignIn}>
          <div className="form-group-auth">
            <input type="email" id="email" name="email" required />
            <label htmlFor="email">Email</label>
          </div>
          <div className="form-group-auth">
            <input type="password" id="password" name="password" required />
            <label htmlFor="password">Password</label>
          </div>
          {errorMessage && <div className="error-message">{errorMessage}</div>}
          <button type="submit">Sign in</button>
        </form>
        <p>Don't you have an account? <a href="/signup">Sign up</a></p>
      </div>
    </div>
  );
};

export default SignIn;