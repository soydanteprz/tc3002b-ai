import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Comparator;
import java.util.StringTokenizer;

public class Comprog {

    static FastReader fr = new FastReader();

    private static void testCase() {
        int n = fr.nextInt();
        int [][] scoreChanges = new int [5][n];
        int [] totalScores = new int [5];

        // initialize
        for (int i = 0; i < 5; i++)
            totalScores[i] = 0;
        
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < n; j++)
                scoreChanges[i][j] = 0;
        
        for (int wordIndex = 0; wordIndex < n; wordIndex++) {
            String nextWord = fr.nextLine();
            for (int charIndex = 0; charIndex < 5; charIndex++) {
                // How many more of the current char ('a' or 'b' or etc.) are in nextWord than
                // the total number of chars in it
                int change = 2 * countCharsInString(nextWord, (char) ('a' + charIndex)) - nextWord.length();
                totalScores[charIndex] += change;
                scoreChanges[charIndex][wordIndex] = change;
            }
        }


        for (int charIndex = 0; charIndex < 5; charIndex++)
            Arrays.sort(scoreChanges[charIndex]);

        int round = 0;
        boolean done = false;
        while (round < n && !done) {
            for (int charIndex = 0; charIndex < 5; charIndex++) {
                if (totalScores[charIndex] > 0) {
                    System.out.println(n - round);
                    done = true;
                    break;
                }
                totalScores[charIndex] -= scoreChanges[charIndex][round];
            }
            round++;
        }
        if (!done)
            System.out.println(0);
    }


    public static int countCharsInString(String str, char c) {
        int cnt = 0;
        for (int i = 0; i < str.length(); i++)
            if (str.charAt(i) == c)
                cnt++;
        return cnt;
    }

    public static void main(String[] args) {
        int t;
        t = fr.nextInt();
        for (int i = 0; i < t; i++)
            testCase();
    }




    

    



    
}
