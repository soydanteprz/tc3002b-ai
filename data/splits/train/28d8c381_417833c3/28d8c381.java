import java.io.*;
import java.sql.SQLOutput;
import java.util.*;

public class Main {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int numCases = Integer.parseInt(br.readLine());
        for (int i = 0; i < numCases; i++) {
            int length = Integer.parseInt(br.readLine());
            int found = 0;
            int[] ret = new int[length + 1];
            for (int j = 1; j <= length; j++) {
                if (found == length - 1) {
                    break;
                }
                if (ret[j] == 0) {
                    System.out.println("? " + j);
                    System.out.flush();
                    int start = Integer.parseInt(br.readLine());
                    int lastNum = start;
                    boolean cont = true;
                    while (cont) {
                        System.out.println("? " + j);
                        System.out.flush();
                        int num = Integer.parseInt(br.readLine());
                        ret[lastNum] = num;
                        found++;
                        lastNum = num;
                        if (num == start)
                            cont = false;
                    }
                }
            }
            for (int j = 0; j <= length; j++)
                if (ret[j] == 0) ret[j] = j;
            System.out.print("! ");
            for (int j = 1; j <= length; j++)
                System.out.print(ret[j] + " ");
            System.out.println();
        }
        br.close();
    }
}
