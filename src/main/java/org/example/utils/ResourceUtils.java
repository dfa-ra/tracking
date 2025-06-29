package org.example.utils;

import java.io.*;

public class ResourceUtils {

    public static File extractToTempFile(String resourcePath) throws IOException {
        InputStream input = ResourceUtils.class.getClassLoader().getResourceAsStream(resourcePath);
        if (input == null) {
            throw new FileNotFoundException("Ресурс не найден: " + resourcePath);
        }

        String extension = getExtension(resourcePath);
        File tempFile = File.createTempFile("resource_", extension);
        tempFile.deleteOnExit();

        try (OutputStream output = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[8192];
            int bytesRead;
            while ((bytesRead = input.read(buffer)) != -1) {
                output.write(buffer, 0, bytesRead);
            }
        }

        return tempFile;
    }

    private static String getExtension(String path) {
        int dot = path.lastIndexOf('.');
        return (dot != -1) ? path.substring(dot) : ".tmp";
    }
}
