import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;


public class C {

    private static StringTokenizer st;
    private static BufferedReader br;
    private static PrintWriter out;

    static boolean[]used;
    static int[]p;

    public static void main(String[] args) throws IOException {
        br = new BufferedReader(new InputStreamReader(System.in));
        out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)));

        int t = nextInt();
        while (t --> 0) {
            int n = nextInt();
            used = new boolean[n+1];
            p = new int[n+1];
            for (int i = 1; i <= n; i++) {
                if (!used[i]) {
                    go(i);
                }
            }
            System.out.print("! ");
            for (int i = 1; i <= n; i++) {
                System.out.print(p[i]+" ");
            }
            System.out.println();
        }
        out.close();
    }

    private static void go(int i) throws IOException {
        List<Integer> inCycle = new ArrayList<>();
        int first = -1;
        while (true) {
            System.out.println("? " + i);
            int x = nextInt();
            used[x] = true;
            if (x == first) {
                break;
            }
            if (first == -1) {
                first = x;
            }
            inCycle.add(x);
        }
        List<Integer> numbers = new ArrayList<>();
        for (int j = 0; j < inCycle.size(); j++) {
            if (inCycle.get(j) == i) {
                for (int k = j; k < inCycle.size(); k++) {
                    numbers.add(inCycle.get(k));
                }
                for (int k = 0; k < j; k++) {
                    numbers.add(inCycle.get(k));
                }
                break;
            }
        }
        if (numbers.size() == 1) {
            p[numbers.get(0)] = numbers.get(0);
        } else {
            for (int j = 0; j < numbers.size()-1; j++) {
                p[numbers.get(j)] = numbers.get(j+1);
            }
            p[numbers.get(numbers.size()-1)] = numbers.get(0);
        }
    }

    private static int nextInt() throws IOException {
        return Integer.parseInt(next());
    }

    

    

    private static String next() throws IOException {
        while (st==null || !st.hasMoreTokens())
            st = new StringTokenizer(br.readLine());
        return st.nextToken();
    }

}
