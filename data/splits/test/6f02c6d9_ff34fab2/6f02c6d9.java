import java.io.*;
import java.util.*;

public class Main {


    static long mod = 1000000007;

    

    

    static InputReader sc = new InputReader(System.in);
    static PrintWriter out = new PrintWriter(System.out);

    public static void main(String[] args) throws IOException {

        int t = sc.nextInt();
        while (t-- > 0) {

            int n = sc.nextInt();
            int[] A = new int[n];

            for (int i = 0; i < A.length; i++) {
                A[i] = sc.nextInt();
            }
            String word = sc.next();
            ArrayList<Integer> blue = new ArrayList<>();
            ArrayList<Integer> red = new ArrayList<>();

            for (int i = 0; i < word.length(); i++) {
                if (word.charAt(i) == 'R') {
                    red.add(A[i]);
                } else {
                    blue.add(A[i]);
                }
            }
            Collections.sort(blue);
            Collections.sort(red);

            boolean possible = true;

            int a = 1;
            for (int i = 0; i < blue.size(); i++, a++) {
                if (blue.get(i) < a) {
                    possible = false;
                    break;
                }
            }

            for (int i = 0; i < red.size(); i++, a++) {
                if (red.get(i) > a) {
                    possible = false;
                    break;
                }
            }

            if (possible) out.println("YES");
            else out.println("NO");


        }
        out.flush();
        out.close();
    }

    

    

    

}
