// SideMenu.js
import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Drawer, List, ListItem, ListItemText, IconButton } from '@material-ui/core';
import MenuIcon from '@material-ui/icons/Menu';
import { supabase } from '../../utils/supabase';

const SideMenu = ({ userEmail }) => {
  const navigate = useNavigate();

  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleSignOut = async () => {
    toggleDrawer(false);
    const { error } = await supabase.auth.signOut();
    if (error) console.log("Error signing out: ", error);
    else {
      toggleDrawer(false);
      navigate('/login');
    }
  };

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setDrawerOpen(open);
  };

  return (
    <>
      <div className="menu-button-left">
        <IconButton edge="start" color="inherit" aria-label="menu" onClick={toggleDrawer(true)} className="menu-button">
          <MenuIcon />
        </IconButton>
      </div>
      <Drawer open={drawerOpen} onClose={toggleDrawer(false)}>
        <List>
          <ListItem>
            <ListItemText primary={userEmail} />
          </ListItem>
          <ListItem button key="Closet" component={Link} to="/closet" onClick={toggleDrawer(false)}>
            <ListItemText primary="Closet" />
          </ListItem>
          <ListItem button key="Favourites" component={Link} to="/favourites" onClick={toggleDrawer(false)}>
            <ListItemText primary="Favourites" />
          </ListItem>
          <ListItem button key="Upload" component={Link} to="/upload" onClick={toggleDrawer(false)}>
            <ListItemText primary="Upload" />
          </ListItem>
          <ListItem button key="Sign out" onClick={handleSignOut}>
            <ListItemText primary="Sign out" />
          </ListItem>
        </List>
      </Drawer>
    </>
  );
};

export default SideMenu;
