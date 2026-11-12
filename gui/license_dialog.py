"""
LiDAR Workbench — User Account & License Dialog.
"""

from __future__ import annotations

import logging
from typing import Optional
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QMessageBox,
    QApplication,
    QTabWidget,
    QWidget,
)
from ..licensing import get_license_manager, DEFAULT_SERVER_URL

logger = logging.getLogger("lidar_workbench.gui.license_dialog")


class LicenseDialog(QDialog):
    """Dialog for user sign in, viewing HWID, quota status, and key activation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LiDAR Workbench — Account & License")
        self.setMinimumWidth(540)
        self._lm = get_license_manager()
        self._build_ui()
        self._refresh_info()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # ── Current Status Group ──────────────────────────────────────
        status_group = QGroupBox("Account Status & Quota")
        status_layout = QVBoxLayout(status_group)

        self._user_label = QLabel("User: Not signed in (Local Free Mode)")
        self._user_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #38bdf8;")
        status_layout.addWidget(self._user_label)

        self._plan_label = QLabel("Plan: Free Community")
        self._plan_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        status_layout.addWidget(self._plan_label)

        self._usage_label = QLabel("Projects: 0 / 1 used")
        status_layout.addWidget(self._usage_label)

        hwid_layout = QHBoxLayout()
        hwid_lbl = QLabel("Machine HWID:")
        self._hwid_val = QLineEdit(self._lm.hwid)
        self._hwid_val.setReadOnly(True)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy_hwid)
        hwid_layout.addWidget(hwid_lbl)
        hwid_layout.addWidget(self._hwid_val)
        hwid_layout.addWidget(copy_btn)
        status_layout.addLayout(hwid_layout)

        self._logout_btn = QPushButton("Sign Out / Reset")
        self._logout_btn.clicked.connect(self._on_logout)
        self._logout_btn.setVisible(False)
        status_layout.addWidget(self._logout_btn)

        layout.addWidget(status_group)

        # ── Tabs: Sign In vs License Key ──────────────────────────────
        self._tabs = QTabWidget()

        # Tab 1: Account Sign In
        sign_in_tab = QWidget()
        sign_in_layout = QVBoxLayout(sign_in_tab)
        
        email_row = QHBoxLayout()
        email_row.addWidget(QLabel("Email:"))
        self._email_input = QLineEdit()
        self._email_input.setPlaceholderText("user@company.com")
        email_row.addWidget(self._email_input)
        sign_in_layout.addLayout(email_row)

        pwd_row = QHBoxLayout()
        pwd_row.addWidget(QLabel("Password:"))
        self._pwd_input = QLineEdit()
        self._pwd_input.setEchoMode(QLineEdit.Password)
        pwd_row.addWidget(self._pwd_input)
        sign_in_layout.addLayout(pwd_row)

        server_row = QHBoxLayout()
        server_row.addWidget(QLabel("Server URL:"))
        self._server_input = QLineEdit(DEFAULT_SERVER_URL)
        server_row.addWidget(self._server_input)
        sign_in_layout.addLayout(server_row)

        self._sign_in_btn = QPushButton("Sign In with Company Account")
        self._sign_in_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._sign_in_btn.clicked.connect(self._on_sign_in)
        sign_in_layout.addWidget(self._sign_in_btn)

        portal_links = QHBoxLayout()
        reg_btn = QPushButton("Create New Account")
        reg_btn.clicked.connect(self._open_register_portal)
        buy_btn = QPushButton("Buy License / Pricing")
        buy_btn.clicked.connect(self._open_pricing_portal)
        portal_links.addWidget(reg_btn)
        portal_links.addWidget(buy_btn)
        sign_in_layout.addLayout(portal_links)

        self._tabs.addTab(sign_in_tab, "Account Sign In")

        # Tab 2: License Key
        key_tab = QWidget()
        key_layout = QVBoxLayout(key_tab)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("License Key:"))
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("LW-XXXX-XXXX-XXXX-XXXX")
        if self._lm.license_key:
            self._key_input.setText(self._lm.license_key)
        key_row.addWidget(self._key_input)
        key_layout.addLayout(key_row)

        server_row2 = QHBoxLayout()
        server_row2.addWidget(QLabel("Server URL:"))
        self._server_input2 = QLineEdit(DEFAULT_SERVER_URL)
        server_row2.addWidget(self._server_input2)
        key_layout.addLayout(server_row2)

        self._activate_btn = QPushButton("Activate License Key")
        self._activate_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self._activate_btn.clicked.connect(self._on_activate)
        key_layout.addWidget(self._activate_btn)

        self._tabs.addTab(key_tab, "Enter License Key")

        layout.addWidget(self._tabs)

        # ── Close Button ──────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _refresh_info(self):
        if self._lm.is_logged_in:
            self._user_label.setText(f"👤 Signed In: {self._lm.user_email}")
            self._logout_btn.setVisible(True)
        else:
            self._user_label.setText("👤 Not signed in (Local Free Mode)")
            self._logout_btn.setVisible(False)

        plan_name = self._lm.plan.replace("_", " ").title()
        self._plan_label.setText(f"Plan: {plan_name}")
        allowed, current, limit = self._lm.can_create_project()
        limit_str = "Unlimited" if limit == -1 else str(limit)
        self._usage_label.setText(f"Projects: {current} / {limit_str} used")

    def _open_register_portal(self):
        url = self._server_input.text().strip().rstrip("/") + "/register"
        QDesktopServices.openUrl(QUrl(url))

    def _open_pricing_portal(self):
        url = self._server_input.text().strip().rstrip("/") + "/pricing"
        QDesktopServices.openUrl(QUrl(url))

    def _copy_hwid(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self._hwid_val.text())
        QMessageBox.information(self, "Copied", "Hardware ID copied to clipboard.")

    def _on_logout(self):
        self._lm.logout()
        self._refresh_info()
        QMessageBox.information(self, "Signed Out", "You have been signed out.")

    def _on_sign_in(self):
        email = self._email_input.text().strip()
        pwd = self._pwd_input.text().strip()
        server_url = self._server_input.text().strip()

        if not email or not pwd:
            QMessageBox.warning(self, "Missing Fields", "Please enter your email and password.")
            return

        self._sign_in_btn.setEnabled(False)
        self._sign_in_btn.setText("Signing in...")
        QApplication.processEvents()

        success, msg = self._lm.login_online(email, pwd, server_url=server_url)

        self._sign_in_btn.setEnabled(True)
        self._sign_in_btn.setText("Sign In with Company Account")

        if success:
            QMessageBox.information(self, "Success", msg)
            self._refresh_info()
        else:
            QMessageBox.critical(self, "Login Error", msg)

    def _on_activate(self):
        key = self._key_input.text().strip()
        server_url = self._server_input2.text().strip()
        if not key:
            QMessageBox.warning(self, "Invalid Key", "Please enter a valid license key.")
            return

        self._activate_btn.setEnabled(False)
        self._activate_btn.setText("Connecting...")
        QApplication.processEvents()

        success, msg = self._lm.activate_online(key, server_url=server_url)

        self._activate_btn.setEnabled(True)
        self._activate_btn.setText("Activate License Key")

        if success:
            QMessageBox.information(self, "Success", msg)
            self._refresh_info()
        else:
            QMessageBox.critical(self, "Activation Error", msg)
