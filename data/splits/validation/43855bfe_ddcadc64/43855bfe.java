import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

public class O4 {
    private static String solve(int n, int[] a) {

        if (n == 2) {
            return a[1] + " " + (-a[0]);
        }
        int res[] = new int[n];
        if (n % 2 == 0) {
            diagonalAlignment(a, res, n);
        } else {

            diagonalAlignment(a, res, n - 3);
            if ( a[n - 3] + a[n - 2] != 0)
            {
                res[n - 1] = a[n - 3] + a[n - 2];
                res[n - 2] = -a[n - 1];
                res[n - 3] = -a[n - 1];
            }

            else if (a[n - 1] + a[n - 2] != 0){
                res[n - 3] = a[n - 1] + a[n - 2];
                res[n - 2] = -a[n - 3];
                res[n - 1] = -a[n - 3];
            }
            else {
                res[n - 2] = a[n - 1] + a[n - 3];
                res[n - 1] = -a[n - 2];
                res[n - 3] = -a[n - 2];

            }
        }


        return formattedArray(res);

    }

    private static void diagonalAlignment(int[] a, int[] res, int n) {


        for (int i = 0; i < n; i += 2) {
            res[i + 1] = -a[i];
            res[i] = a[i + 1];
        }


    }


    public static void main(String[] args)
            throws IOException {
        Scanner s = new Scanner();
        int t = 1;
        t = s.nextInt();
        StringBuilder ans = new StringBuilder();
        int count = 0;
        while (t-- > 0) {
            int n = s.nextInt();

            int a[] = new int[n];
            getInputs(s, a, n);


            ans.append(solve(n, a)).append("\n");
        }
        System.out.println(ans.toString());
    }

    

    

    

    

    

    

    public static String formattedArray(int a[]) {
        StringBuilder res = new StringBuilder("");
        for (int e : a)
            res.append(e).append(" ");
        return res.toString().trim();

    }

    private static void getInputs(Scanner s, int[] a, int n) throws IOException {
        for (int i = 0; i < n; i++) {
            a[i] = s.nextInt();
        }
    }
}
