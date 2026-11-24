; Inno Setup Script for LiDAR Workbench
; Target: Windows 10/11 x64 (RTX 30, 40, 50 Series Support)

#define MyAppName "LiDAR Workbench"
#define MyAppVersion "1.0"
#define MyAppPublisher "Kabelik GmbH"
#define MyAppURL "https://kabelik.at"
#define MyAppExeName "LiDAR-Workbench.exe"
#define MyAppDir "dist\LiDAR-Workbench"

[Setup]
AppId={{E68A9F21-6A72-4B43-98C1-87F95A1E1128}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
SetupIconFile=gui\assets\app_icon.ico
UninstallIconFile=gui\assets\app_icon.ico
OutputDir=dist
OutputBaseFilename=LiDAR_Workbench_Setup_v1.0_RTX30_40_50
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DiskSpanning=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#MyAppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall
