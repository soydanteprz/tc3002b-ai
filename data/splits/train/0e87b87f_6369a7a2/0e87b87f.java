import java.io.*;
import java.util.*;
public class table {
    
    public static void main(String[] args) throws java.io.IOException {
        File file = new File("hoofball.in"); // change this
        FastReader input = new FastReader(); // change this
        PrintWriter output = new PrintWriter("hoofball.out"); // change this or ph

        int testcases = input.nextInt();
        for (int t = 0; t < testcases; t++) {
            int rows = input.nextInt();
            int columns = input.nextInt();
            char[][] matrix = new char[rows][columns];
            for (int i = 0; i < rows; i++) {
                matrix[i] = input.next().toCharArray();
            }
            int counter = 0;
            ArrayList<String> list = new ArrayList<String>();
            for (int r = 0; r < rows - 1; r++) {
                for (int c = 0; c < columns - 1; c++) {
                    int r1 = r + 1;
                    int c1 = c + 1;
                    int r2 = r + 2;
                    int c2 = c + 2;
                    while (matrix[r][c] == '1' || matrix[r+1][c] == '1' || matrix[r][c+1] == '1' || matrix[r+1][c+1] == '1') {
                        if (matrix[r][c] == '0' && matrix[r+1][c] == '1' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '1') { // 0 1 1 1
                            list.add(r2 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r+1][c] = '0';
                            matrix[r][c+1] = '0';
                            matrix[r+1][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '0' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '1') { // 1 0 1 1
                            list.add(r1 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '0';
                            matrix[r][c+1] = '0';
                            matrix[r+1][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '1' && matrix[r][c+1] == '0' && matrix[r+1][c+1] == '1') { // 1 1 0 1
                            list.add(r1 +" "+ c1 +" "+ r2 +" "+ c1 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '0';
                            matrix[r+1][c] = '0';
                            matrix[r+1][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '1' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '0') { // 1 1 1 0
                            list.add(r1 +" "+ c1 +" "+ r2 +" "+ c1 +" "+ r1 +" "+ c2);
                            matrix[r][c] = '0';
                            matrix[r+1][c] = '0';
                            matrix[r][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '1' && matrix[r][c+1] == '0' && matrix[r+1][c+1] == '0') { // 1 1 0 0
                            list.add(r1 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '0';
                            matrix[r][c+1] = '1';
                            matrix[r+1][c+1] = '1';
                        } else if (matrix[r][c] == '0' && matrix[r+1][c] == '0' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '1') { // 0 0 1 1
                            list.add(r1 +" "+ c1 +" "+ r2 +" "+ c1 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '1';
                            matrix[r+1][c] = '1';
                            matrix[r+1][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '0' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '0') { // 1 0 1 0
                            list.add(r1 +" "+ c1 +" "+ r2 +" "+ c1 +" "+ r1 +" "+ c2);
                            matrix[r][c] = '0';
                            matrix[r+1][c] = '1';
                            matrix[r][c+1] = '0';
                        } else if (matrix[r][c] == '0' && matrix[r+1][c] == '1' && matrix[r][c+1] == '0' && matrix[r+1][c+1] == '1') { // 0 1 0 1
                            list.add(r1 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '1';
                            matrix[r][c+1] = '1';
                            matrix[r+1][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '0' && matrix[r][c+1] == '0' && matrix[r+1][c+1] == '0') { // 1 0 0 0
                            list.add(r1 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '0';
                            matrix[r][c+1] = '1';
                            matrix[r+1][c+1] = '1';
                        } else if (matrix[r][c] == '0' && matrix[r+1][c] == '1' && matrix[r][c+1] == '0' && matrix[r+1][c+1] == '0') { // 0 1 0 0
                            list.add(r2 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r+1][c] = '0';
                            matrix[r][c+1] = '1';
                            matrix[r+1][c+1] = '1';
                        } else if (matrix[r][c] == '0' && matrix[r+1][c] == '0' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '0') { // 0 0 1 0
                            list.add(r2 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r+1][c] = '1';
                            matrix[r][c+1] = '0';
                            matrix[r+1][c+1] = '1';
                        } else if (matrix[r][c] == '0' && matrix[r+1][c] == '0' && matrix[r][c+1] == '0' && matrix[r+1][c+1] == '1') { // 0 0 0 1
                            list.add(r1 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '1';
                            matrix[r][c+1] = '1';
                            matrix[r+1][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '1' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '1') { // 1 1 1 1
                            list.add(r2 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c2);
                            matrix[r+1][c] = '0';
                            matrix[r][c+1] = '0';
                            matrix[r+1][c+1] = '0';
                        } else if (matrix[r][c] == '1' && matrix[r+1][c] == '0' && matrix[r][c+1] == '0' && matrix[r+1][c+1] == '1') { // 1 0 0 1
                            list.add(r1 +" "+ c1 +" "+ r1 +" "+ c2 +" "+ r2 +" "+ c1);
                            matrix[r][c] = '0';
                            matrix[r][c+1] = '1';
                            matrix[r+1][c] = '1';
                        } else if (matrix[r][c] == '0' && matrix[r+1][c] == '1' && matrix[r][c+1] == '1' && matrix[r+1][c+1] == '0') { // 0 1 1 0
                            list.add(r1 +" "+ c1 +" "+ r2 +" "+ c1 +" "+ r2 +" "+ c2);
                            matrix[r][c] = '1';
                            matrix[r+1][c] = '0';
                            matrix[r+1][c+1] = '1';
                        } else {
                            System.out.println("error");
                        }
                        counter++;
                    }
                }
            }
            System.out.println(counter);
            for (int i = 0; i < list.size(); i++) {
                System.out.println(list.get(i));
            }
        }

        output.close();
    }
}
