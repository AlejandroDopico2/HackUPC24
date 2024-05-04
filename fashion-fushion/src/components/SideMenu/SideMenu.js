// SideMenu.js
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Drawer, List, ListItem, ListItemText, IconButton } from '@material-ui/core';
import MenuIcon from '@material-ui/icons/Menu';

const SideMenu = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
      return;
    }
    setDrawerOpen(open);
  };

  return (
    <>
      <div className="menu-button-left"> {/* Añadido un div contenedor */}
        <IconButton edge="start" color="inherit" aria-label="menu" onClick={toggleDrawer(true)} className="menu-button">
          <MenuIcon />
        </IconButton>
      </div>
      <Drawer open={drawerOpen} onClose={toggleDrawer(false)}>
        <List>
          <ListItem button key="Closet" component={Link} to="/closet" onClick={toggleDrawer(false)}>
            <ListItemText primary="Closet" />
          </ListItem>
          <ListItem button key="Favourites" component={Link} to="/favourites" onClick={toggleDrawer(false)}>
            <ListItemText primary="Favourites" />
          </ListItem>
          <ListItem button key="Upload" component={Link} to="/upload" onClick={toggleDrawer(false)}>
            <ListItemText primary="Upload" />
          </ListItem>
        </List>
      </Drawer>
    </>
  );
};

export default SideMenu;
