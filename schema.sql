-- Created automatically on startup; run manually only if you prefer.
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL, sex TINYINT NOT NULL, cp TINYINT NOT NULL,
    trestbps INT NOT NULL, chol INT NOT NULL, fbs TINYINT NOT NULL,
    restecg TINYINT NOT NULL, thalach INT NOT NULL, exang TINYINT NOT NULL,
    oldpeak FLOAT NOT NULL, slope TINYINT NOT NULL, ca TINYINT NOT NULL,
    thal TINYINT NOT NULL, prob_logreg FLOAT NOT NULL, prob_svm FLOAT NOT NULL,
    flagged TINYINT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
