import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Menorah {
    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int t = Integer.parseInt(br.readLine());
        while (t > 0) {
            int n = Integer.parseInt(br.readLine());
            String source = br.readLine();
            String destination = br.readLine();
            int sameStatusOnes = 0;
            int sameStatusZeros = 0;
            int diffStatusOnes = 0;
            int diffStatusZeros = 0;
            for (int i = 0; i < n; i++) {
                char c1 = source.charAt(i);
                char c2 = destination.charAt(i);
                if (c1 == c2) {
                    if (c1 == '0') {
                        sameStatusZeros += 1;
                    } else {
                        sameStatusOnes += 1;
                    }
                } else {
                    if (c1 == '0') {
                        diffStatusZeros += 1;
                    } else {
                        diffStatusOnes += 1;
                    }
                }
            }
            int sameStatus = sameStatusOnes + sameStatusZeros;
            int diffStatus = diffStatusOnes + diffStatusZeros;
            //first case
            if (sameStatus == n) {
                System.out.println(0);
            } else if (diffStatus == n) {
                //second case
                if (diffStatus % 2 == 0 && diffStatusOnes == (n + 1) / 2)
                    System.out.println(n);
                else
                    System.out.println(-1);
            } else {
                int op1 = -1;
                int op2 = -1;
                if (sameStatus % 2 != 0 && sameStatusOnes == (sameStatus + 1) / 2)
                    op1 = sameStatus;
                if (diffStatus % 2 == 0 && diffStatusOnes == (diffStatus + 1) / 2)
                    op2 = diffStatus;

                if (op1 != -1 && op2 != -1)
                    System.out.println(Integer.min(op1, op2));
                else if (op1 != -1)
                    System.out.println(op1);
                else if (op2 != -1)
                    System.out.println(op2);
                else
                    System.out.println(-1);
            }
            t--;
        }
    }
}
