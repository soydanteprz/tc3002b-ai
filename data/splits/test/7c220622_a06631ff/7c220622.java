import java.io.*;
import java.util.*;

public class D {
    public static void main(String[] args) throws IOException {
        in.init();
        int n = in.readInt();
        for (int i = 0; i < n; i++) {
            sol();
        }
        in.dinit();
    }

    private static void sol() throws IOException {
        // ArrayList<Integer> arr = new ArrayList<>();
        int n = in.readInt();

        int arr[] = in.arr(n);
        int three2n = 1;
        for (int i = 0; i < arr.length; i++) {
            three2n *= 3;
        }
        for (int i = 1; i < three2n; i++) {
            int sum = i;
            int actualsum = 0;
            for (int j = 0; j < arr.length; j++) {
                int change = sum % 3;
                change = (change == 2) ? -1 : change;
                actualsum += change * arr[j];
                sum = sum / 3;
            }
            if (actualsum == 0) {
                System.out.println("YES");
                return;
            }

        }
        System.out.println("NO");

    }

}


class in {

    static BufferedReader br;
    static PrintWriter out;
    static StringTokenizer st;

    public static void init() throws IOException {
         br = new BufferedReader(new InputStreamReader(System.in));
         out = new PrintWriter(new OutputStreamWriter(System.out));
       // br = new BufferedReader(new FileReader("in.txt"));
        //out = new PrintWriter(new FileWriter("out.txt"));

        // out.close();
    }

    static int[] arr(int n) throws IOException {
        int arr[] = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = in.readInt();

        }
        return arr;

    }

    public static void dinit() throws IOException {
        br.close();

        out.close();
    }

    static String next() throws IOException {
        while (st == null || !st.hasMoreTokens())
            st = new StringTokenizer(br.readLine().trim());
        return st.nextToken();
    }

    

    static int readInt() throws IOException {
        return Integer.parseInt(next());
    }

    

    

    static String readLine() throws IOException {
        return br.readLine().trim();
    }
}
