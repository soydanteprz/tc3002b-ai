import java.io.*;
import java.util.*;

public class C {

    public static void main(String[] args) {
        scan = new MyScanner();
        out = new PrintWriter(new BufferedOutputStream(System.out));

        int t = scan.nextInt();
        while (t-->0){
            solve();
        }

        out.close();
    }

    static void solve() {
        int n = scan.nextInt();
        int[][] words = new int[n][6];
        for (int i = 0; i < n; i++) {
            String input = scan.next();
            words[i][0] = input.length();
            for (int j = 0; j < input.length(); j++) {
                words[i][input.charAt(j)-96]++;
            }
            for (int j = 1; j <= 5; j++) {
                words[i][j] =  words[i][0] - 2*words[i][j];
            }
        }
        int maxCounter = 0;
        for (int i = 1; i <= 5; i++) {
            int[] arr = new int[n];

            for (int j = 0; j < n; j++) {
                arr[j] = words[j][i];
            }
            shuffleArray(arr);
            Arrays.sort(arr);
            int sum = 0;
            int counter = 0;
            while (counter < n && sum + arr[counter] < 0){
                sum += arr[counter];
                counter++;
            }
            maxCounter = Math.max(maxCounter, counter);
        }
        out.println(maxCounter);
    }

    private static void shuffleArray(int[] array)
    {
        int index, temp;
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--)
        {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    public static MyScanner scan;
    public static PrintWriter out;

    
}
