import React, { useState } from 'react';
import { supabase } from '../../utils/supabase';
import { useNavigate } from 'react-router-dom';

const SignUp = () => {
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();

  const handleEmailSignUp = async (event) => {
    event.preventDefault();
    const { data, error } = await supabase.auth.signUp({
      email: event.target.email.value,
      password: event.target.password.value,
    });
    if (error) {
      setErrorMessage(error.message);
      console.error('Error signing up:', error.message);
    } else {
      console.log('User signed up:', data.user);
      navigate('/login');
    }
  };

  return (
    <div className="form-container">
      <div className="form-form">
        <h2>Create account</h2>
        <form onSubmit={handleEmailSignUp}>
          <div className="form-group-auth">
            <input type="email" id="email" name="email" required />
            <label htmlFor="email">Email</label>
          </div>
          <div className="form-group-auth">
            <input type="password" id="password" name="password" required />
            <label htmlFor="password">Password</label>
          </div>
          {errorMessage && <div className="error-message">{errorMessage}</div>}
          <button type="submit">Crate account</button>
        </form>
        <p>Already have an account? <a href="/login">Log in</a></p>
      </div>
    </div>
  );
};

export default SignUp;