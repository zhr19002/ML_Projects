import numpy as np

def add_wind_noise(df):
    df = df.copy()
    eps_e, eps_n = np.zeros(len(df)), np.zeros(len(df))
    
    # Parameters
    phi = 0.9       # AR(1) coefficient
    rel_err = 0.15  # 15% RMS forecast error

    # Error std proportional to total wind speed
    sigma = rel_err * df['u'].values / np.sqrt(2)

    # Initialize from stationary distribution
    eps_e[0] = np.random.normal(0, sigma[0])
    eps_n[0] = np.random.normal(0, sigma[0])

    # Generate AR(1) forecast errors
    for i in range(1, len(df)):
        innov_std = sigma[i] * np.sqrt(1 - phi**2)
        eps_e[i] = phi * eps_e[i-1] + np.random.normal(0, innov_std)
        eps_n[i] = phi * eps_n[i-1] + np.random.normal(0, innov_std)

    # Add forecast errors
    df['u_e'] = df['u_e'] + eps_e
    df['u_n'] = df['u_n'] + eps_n
    df['u'] = np.sqrt(df['u_e']**2 + df['u_n']**2)

    return df