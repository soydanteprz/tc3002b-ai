import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.StringTokenizer;

public class D {

    static PrintWriter out = new PrintWriter(System.out);

    

    public static void main(String[] args) {
        FastScanner sc = new FastScanner();
        int t = sc.nextInt();
        while (t-- > 0) {
            int n = sc.nextInt();
            Node head = null;
            boolean notPossible = false;
            for (int i = 0; i < n; i++) {
                int x = sc.nextInt();
                if (head == null) {
                    head = new Node(x);
                } else {
                    if ((head.next != null && x > head.next.value) || (head.prev != null && x < head.prev.value)) {
                        notPossible = true;
                    } else if ((head.next == null || x <= head.next.value) && x > head.value) {
                        if (head.next != null && x == head.next.value) {
                            head = head.next;
                            continue;
                        }
                        Node temp = head.next;
                        Node next = new Node(x);
                        head.next = next;
                        next.prev = head;
                        next.next = temp;
                        if (temp != null) {
                            temp.prev = next;
                        }
                        head = next;
                    } else if ((head.prev == null || x >= head.prev.value) && x < head.value) {
                        if (head.prev != null && x == head.prev.value) {
                            head = head.prev;
                            continue;
                        }
                        Node temp = head.prev;
                        Node prev = new Node(x);
                        head.prev = prev;
                        prev.next = head;
                        prev.prev = temp;
                        if (temp != null) {
                            temp.next = prev;
                        }
                        head = prev;
                    }
                }
            }
            if (notPossible) {
                out.println("NO");
            } else {
                out.println("YES");

            }        }
        out.close();
    }

    static class Node {
        int value;
        Node prev;
        Node next;
        Node(int value) {
            this.value = value;
            this.prev = null;
            this.next = null;
        }
    }
}
